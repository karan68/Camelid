import { useEffect, useState } from 'react'
import { SUPPORTED_MODELS } from '../../lib/supportedModels'
import { formatBytes } from '../../lib/formatters'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { IconCheck, IconDownload, IconStop } from '../ui/icons'

/* "Supported models" — the curated rows Camelid can download + run, with a
   one-click Download wired to the existing catalog install + progress tracking. */
export function SupportedModels({
  models = [],
  runtime,
  apiBase = '',
  installCatalogModel,
  cancelModelDownload,
  activateModel,
  loadingModelId = '',
}) {
  const online = runtime?.status === 'online'

  // "Downloaded" must mean the GGUF is actually on disk right now — verify against
  // the backend's live models/ scan (/api/models/local), never a persisted client
  // record alone. A localStorage record (id === catalog_id) lingers after the file
  // is deleted or after a half-finished download, which would otherwise make the
  // card show "Downloaded" instantly without the file. Mirrors CatalogLaneBrowse.
  const base = (apiBase || '').replace(/\/$/, '')
  const [localNames, setLocalNames] = useState(new Set())
  useEffect(() => {
    if (!online) return undefined
    let cancelled = false
    const refresh = async () => {
      try {
        const res = await fetch(`${base}/api/models/local`)
        if (!res.ok) return
        const body = await res.json()
        if (!cancelled) setLocalNames(new Set((body.models || []).map((m) => m.filename)))
      } catch {
        /* keep the previous snapshot on a transient error */
      }
    }
    refresh()
    const timer = setInterval(refresh, 4000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [base, online])

  return (
    <section className="supported-models" aria-label="Supported models you can download">
      <header className="supported-models__head">
        <div>
          <p className="supported-models__kicker">Supported models</p>
          <h3 className="supported-models__title">Download a model Camelid can run</h3>
        </div>
        {!online && <Chip tone="warn" dot>Start the backend to download</Chip>}
      </header>

      <div className="supported-models__grid">
        {SUPPORTED_MODELS.map((item) => {
          const tracked = models.find((m) => m.id === item.catalog_id)
          const status = tracked?.status
          const downloading = status === 'downloading' || status === 'canceling'
          const progress = Math.max(0, Math.min(100, Number(tracked?.progress) || 0))
          // Authoritative: the GGUF is present in models/ on disk right now.
          const present = localNames.has(item.filename)
          const downloaded = present && !downloading
          const loadedNow = Boolean(tracked?.loaded_now)
          const loadId = tracked?.id || item.catalog_id
          const busy = loadingModelId === loadId

          return (
            <article key={item.catalog_id} className={`supported-model ${item.recommended ? 'is-recommended' : ''}`}>
              <div className="supported-model__top">
                <h4 className="supported-model__name">{item.name}</h4>
                <div className="supported-model__tags">
                  <Chip tone="neutral">{item.quant}</Chip>
                  <Chip tone="neutral">{formatBytes(item.size_bytes)}</Chip>
                  {item.recommended && <Chip tone="accent">Recommended</Chip>}
                </div>
              </div>
              <p className="supported-model__blurb">{item.blurb}</p>

              {downloading ? (
                <div className="supported-model__progress-row">
                  <div className="supported-model__progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
                    <span style={{ width: `${progress}%` }} />
                  </div>
                  <span className="supported-model__progress-label">{status === 'canceling' ? 'Canceling…' : `${progress}%`}</span>
                  <Button variant="ghost" size="sm" icon={<IconStop size={15} />} onClick={() => cancelModelDownload(tracked.id)} disabled={status === 'canceling'}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <div className="supported-model__actions">
                  {loadedNow ? (
                    <Chip tone="ready" icon={<IconCheck size={15} />}>Loaded</Chip>
                  ) : downloaded ? (
                    <>
                      <Chip tone="ready" icon={<IconCheck size={15} />}>Downloaded</Chip>
                      <Button variant="primary" size="sm" loading={busy} onClick={() => activateModel(loadId)}>
                        Load
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="primary"
                      size="sm"
                      icon={<IconDownload size={16} />}
                      onClick={() => installCatalogModel(item)}
                      disabled={!online}
                      title={online ? `Download ${item.name}` : 'Start the Camelid backend first (Settings)'}
                    >
                      Download
                    </Button>
                  )}
                </div>
              )}
            </article>
          )
        })}
      </div>
    </section>
  )
}

export default SupportedModels
