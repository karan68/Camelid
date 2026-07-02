import { useEffect, useMemo, useState } from 'react'
import { ModelInspector } from '../components/models/ModelInspector'
import { TokenizerPlayground } from '../components/models/TokenizerPlayground'
import { ActiveModelBar } from '../components/models/ActiveModelBar'
import { CatalogLaneBrowse } from '../components/models/CatalogLaneBrowse'
import { DownloadsPanel } from '../components/models/DownloadsPanel'
import { UnsupportedBlocker } from '../components/models/UnsupportedBlocker'
import { Section, SupportedRow, CompatibleRow, EligibleRow, NotAnchoredRow } from '../components/models/LaneRows'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { bucketByLane } from '../lib/modelLanes'
import { IconModels } from '../components/ui/icons'

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
  // Typed fail-closed blocker from a pre-load inspect ({ code, message }), shown
  // verbatim instead of attempting a multi-GB load that cannot run.
  const [blocker, setBlocker] = useState(null)
  const [laneError, setLaneError] = useState('')
  const [cancelingDownloads, setCancelingDownloads] = useState(new Set())
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [importing, setImporting] = useState(false)

  const laneBuckets = useMemo(
    () => (spine.local ? bucketByLane(spine.local.models, capabilities) : null),
    [spine.local, capabilities],
  )
  const activeEntry = useMemo(
    () => spine.local?.models.find((m) => m.filename === spine.activeFilename) || null,
    [spine.local, spine.activeFilename],
  )
  const experimentalRows = laneBuckets
    ? [...laneBuckets.compatible, ...laneBuckets.eligible, ...laneBuckets.not_anchored]
    : []

  // Load a local model into the chat backend. First predict the lane with a
  // header-only inspect (no multi-GB read): if the architecture is not implemented,
  // surface the exact typed blocker and stop — never attempt to run it. Implemented
  // architectures (supported or experimental) load as before.
  const loadModelForChat = async (filename) => {
    setUsingFilename(filename)
    setLaneError('')
    setBlocker(null)
    const path = `${spine.local?.models_dir || 'models'}/${filename}`
    try {
      const inspectRes = await fetch(`${spine.base}/api/models/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      if (inspectRes.ok) {
        const inspect = await inspectRes.json()
        if (inspect?.blocker) {
          setBlocker(inspect.blocker)
          return
        }
      }
      // Implemented (or inspect unavailable) → attempt the real load.
      const res = await fetch(`${spine.base}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: filename, path }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // A typed fail-closed load error (e.g. invalid metadata) becomes a blocker.
        if (body?.error?.code && body.error.code !== 'invalid_model') {
          setBlocker({ code: body.error.code, message: body.error.message })
          return
        }
        throw new Error(body?.error?.message || `load failed (HTTP ${res.status})`)
      }
      await spine.refreshCurrent()
      refreshDashboard?.({ silent: true })
    } catch (err) {
      setLaneError(String(err?.message || err))
    } finally {
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

  const cancelDownloadById = async (id) => {
    setCancelingDownloads((s) => new Set([...s, id]))
    try {
      await spine.cancelDownload(id)
    } finally {
      setCancelingDownloads((s) => {
        const next = new Set(s)
        next.delete(id)
        return next
      })
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
        setLaneError(body?.error?.message || `Smoke-admission did not pass for ${filename}.`)
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
            Load, download, and manage local GGUF models. Section membership is derived live from the
            disk scan and the /api/capabilities support contract.
          </p>
        </div>
        <div className="cxv-head__actions">
          <button
            type="button"
            className="lane-refresh"
            onClick={() => spine.refreshAll()}
            disabled={spine.localLoading}
          >
            {spine.localLoading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </header>

      {/* Zone 1 — active model bar */}
      <ActiveModelBar
        runtime={runtime}
        activeFilename={spine.activeFilename}
        activeEntry={activeEntry}
        capabilities={capabilities}
        busy={unloading || Boolean(loadingModelId)}
        onUnload={handleUnload}
      />
      {laneError ? <p className="lane-error">{laneError}</p> : null}
      {spine.localError && !spine.local ? (
        <p className="lane-empty">Could not list local models: {spine.localError}</p>
      ) : null}

      {/* Zone 2 — supported local models (derived membership only) */}
      <Section
        title="Supported"
        count={laneBuckets ? laneBuckets.supported.length : undefined}
        subtitle="Local models matching an exact supported /api/capabilities row — cross-validated parity."
      >
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : laneBuckets.supported.length ? (
          laneBuckets.supported.map((m) => (
            <SupportedRow
              key={m.filename}
              entry={m}
              active={m.filename === spine.activeFilename}
              busy={usingFilename === m.filename}
              onUse={() => loadModelForChat(m.filename)}
            />
          ))
        ) : (
          <p className="lane-empty">No local model matches a supported row yet — download one below in “Get models”.</p>
        )}
      </Section>

      {/* Zone 3 — everything else local, honestly labeled by evidence state */}
      <Section
        title="Experimental"
        count={laneBuckets ? experimentalRows.length : undefined}
        subtitle="These run without parity anchoring — output is not cross-validated against the reference."
      >
        {blocker ? <UnsupportedBlocker blocker={blocker} className="local-lane-blocker" /> : null}
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : experimentalRows.length ? (
          <>
            {laneBuckets.compatible.map((m) => (
              <CompatibleRow key={m.filename} entry={m} receipt={receipts[m.filename]} />
            ))}
            {laneBuckets.eligible.map((m) => (
              <EligibleRow
                key={m.filename}
                entry={m}
                busy={Boolean(smokeBusy[m.filename])}
                onRun={() => runSmoke(m.filename)}
              />
            ))}
            {laneBuckets.not_anchored.map((m) => (
              <NotAnchoredRow
                key={m.filename}
                entry={m}
                busy={usingFilename === m.filename}
                onUse={() => loadModelForChat(m.filename)}
              />
            ))}
          </>
        ) : (
          <p className="lane-empty">Nothing experimental on disk — every local model matches a supported row.</p>
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
        apiBase={catalogApiBase || apiBase}
        capabilities={capabilities}
        localFilenames={spine.localFilenames}
        downloads={spine.downloads}
        installAvailable={runtimeOnline && catalogInstallAvailable}
        installBlockedReason={
          !runtimeOnline
            ? 'The runtime is offline — start the Camelid backend to download models.'
            : 'The backend does not advertise a catalog-install capability, so downloads stay disabled.'
        }
        onInstallStarted={spine.kickDownloadsPoll}
        onAcquired={() => spine.refreshLocal()}
      />

      {/* Diagnostics — operator tools, collapsed by default. Import-by-path lives
          here because it is the only way to load a GGUF stored outside models/. */}
      <details className="models-diagnostics">
        <summary>Diagnostics</summary>
        <div className="models-diagnostics__body">
          <div className="models-diagnostics__tools">
            <button
              type="button"
              className="lane-refresh"
              onClick={() => setInspectorOpen(true)}
              title="GGUF metadata, tokenizer, tensors — descriptive only, not support evidence"
            >
              Inspect loaded model metadata
            </button>
          </div>

          <div className="models-diagnostics__import">
            <h3>Import a GGUF by path</h3>
            <p className="lane-empty">
              Models inside the <code>models/</code> folder appear above automatically. Use this only
              for a GGUF stored elsewhere; it loads immediately and support still comes from
              /api/capabilities, not filename optimism.
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
              <button
                type="button"
                className="lane-row-action"
                onClick={importFromPath}
                disabled={importing || Boolean(loadingModelId)}
              >
                {importing || loadingModelId ? 'Loading…' : 'Import and load'}
              </button>
            </div>
          </div>

          <TokenizerPlayground apiBase={catalogApiBase || apiBase} />
        </div>
      </details>

      {inspectorOpen && (
        <ModelInspector apiBase={catalogApiBase || apiBase} onClose={() => setInspectorOpen(false)} />
      )}
    </section>
  )
}
