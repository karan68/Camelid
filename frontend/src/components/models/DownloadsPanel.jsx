import { Button } from '../ui/Button'
import { formatBytes } from '../../lib/formatters'

/* Zone 4 — one global live status area for active downloads. Progress comes ONLY
   from the /api/models/catalog/downloads poll (bytes/total) owned by
   useModelsPageData; there is no client-side download record. The panel renders
   nothing when idle, but its data source (the spine hook) stays mounted with the
   page so status is never lost on navigation. */

function pct(dl) {
  if (!dl.total_bytes) return 0
  return Math.max(0, Math.min(100, Math.round((dl.bytes_downloaded / dl.total_bytes) * 100)))
}

export function DownloadsPanel({ downloads = [], onCancel, cancelingIds = new Set() }) {
  const active = downloads.filter((d) => d.status === 'downloading' || d.status === 'preparing')
  if (!active.length) return null

  return (
    <section className="lane-section downloads-panel" aria-label="Downloads in progress">
      <header className="lane-section-head">
        <h2>
          Downloads <span className="lane-section-count">{active.length}</span>
        </h2>
        <p className="lane-section-sub">Canceling a download removes the partially downloaded file.</p>
      </header>
      <div className="lane-section-body">
        {active.map((dl) => (
          <article
            key={dl.id}
            className="lane-row downloads-row"
            aria-label={`${dl.status === 'preparing' ? 'Preparing Ghost MoE for' : 'Downloading'} ${dl.filename}`}
          >
            <div className="lane-row-head">
              <div className="lane-row-id">
                <span className="lane-row-name">{dl.filename}</span>
                <span className="lane-row-meta">
                  {dl.status === 'preparing' ? (
                    <>Preparing Ghost MoE · repacking experts and reclaiming the full GGUF</>
                  ) : (
                    <>
                      {dl.current_artifact_role === 'vision_projector' ? 'Vision projector' : 'Model'}
                      {dl.artifact_count > 1 ? ` ${dl.artifact_index} of ${dl.artifact_count}` : ''} ·{' '}
                      {formatBytes(dl.bytes_downloaded)} / {dl.total_bytes ? formatBytes(dl.total_bytes) : 'size unknown'}
                      {dl.total_bytes ? ` · ${pct(dl)}%` : ''}
                    </>
                  )}
                </span>
              </div>
              {/* Preparing is not cancelable; the meta line already says so, and a
                  Cancel button relabeled into a status caption reads as broken. */}
              {dl.status === 'preparing' ? null : (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onCancel(dl.id)}
                  loading={cancelingIds.has(dl.id)}
                  disabled={cancelingIds.has(dl.id)}
                >
                  Cancel
                </Button>
              )}
            </div>
            <div
              className="downloads-progress"
              role="progressbar"
              aria-label={`Download progress for ${dl.filename}`}
              aria-valuenow={dl.status === 'preparing' ? 100 : pct(dl)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span style={{ width: `${dl.status === 'preparing' ? 100 : pct(dl)}%` }} />
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default DownloadsPanel
