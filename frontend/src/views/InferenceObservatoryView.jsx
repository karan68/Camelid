/* Inference Observatory — a live, truthful visualization of Camelid
   inference. The canvas is a renderer over the real telemetry stream
   (`/api/telemetry/stream`); it never animates inference that is not
   happening. States:
     - telemetry unavailable  → "Waiting for live Camelid telemetry."
     - connected, idle        → ambient sky + invitation to run an inference
     - connected, running     → the canvas follows real events
     - error                  → the failure is shown and listed in details */

import { useEffect, useState } from 'react'
import InferenceCanvas from '../components/observatory/InferenceCanvas'
import MetricsOverlay from '../components/observatory/MetricsOverlay'
import ProofOverlay from '../components/observatory/ProofOverlay'
import DetailsPanel from '../components/observatory/DetailsPanel'
import { useInferenceTelemetry } from '../hooks/useInferenceTelemetry'
import { CONNECTION } from '../lib/inferenceTelemetry'

const MODES = [
  { id: 'art', label: 'Art' },
  { id: 'engineer', label: 'Engineer' },
  { id: 'proof', label: 'Proof' },
]
const MODE_KEY = 'camelid.observatory.mode'

export default function InferenceObservatoryView({ apiBase }) {
  const store = useInferenceTelemetry(apiBase)
  const [mode, setMode] = useState(() => {
    if (typeof window === 'undefined') return 'art'
    // ?obsmode=engineer|proof|art deep-links a display mode (handy when
    // recording); otherwise the last choice is restored.
    const fromUrl = new URLSearchParams(window.location.search).get('obsmode')
    if (MODES.some((m) => m.id === fromUrl)) return fromUrl
    const saved = window.localStorage.getItem(MODE_KEY)
    return MODES.some((m) => m.id === saved) ? saved : 'art'
  })
  const [presentation, setPresentation] = useState(false)
  const [detailsCollapsed, setDetailsCollapsed] = useState(false)

  useEffect(() => {
    if (typeof window !== 'undefined') window.localStorage.setItem(MODE_KEY, mode)
  }, [mode])

  // Presentation mode: Esc leaves it, so the hidden controls stay reachable.
  useEffect(() => {
    if (!presentation) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') setPresentation(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [presentation])

  const connection = store.getConnection()
  const run = store.getRun()
  const live = connection === CONNECTION.LIVE
  const running = live && run.active && !store.isRunStale()
  const errored = run.phase === 'error' || (run.finish && run.finish.status === 'error')

  let statusText = null
  if (!live) {
    statusText = 'Waiting for live Camelid telemetry.'
  } else if (!running && !errored) {
    statusText = 'Start a local inference to watch Camelid work.'
  } else if (errored && !running) {
    statusText = 'Inference failed — see details.'
  }

  return (
    <div className={`observatory ${presentation ? 'is-presentation' : ''}`} data-mode={mode}>
      <InferenceCanvas store={store} showLabels={mode === 'engineer' && !presentation} />

      {statusText && (
        <div className={`observatory-status ${!live ? 'is-waiting' : errored ? 'is-error' : 'is-idle'}`}>
          <span className="observatory-status__dot" aria-hidden="true" />
          {statusText}
        </div>
      )}

      {mode === 'engineer' && <MetricsOverlay store={store} />}
      {mode === 'proof' && <ProofOverlay store={store} />}

      {!presentation && (
        <header className="observatory-toolbar">
          <div className="observatory-toolbar__title">
            <h1>Inference Observatory</h1>
            <span className={`observatory-live-badge ${live ? 'is-live' : ''}`}>
              {live ? (running ? 'live · inference' : 'live') : 'no telemetry'}
            </span>
          </div>
          <div className="observatory-toolbar__controls">
            <div className="observatory-mode-switch" role="tablist" aria-label="Display mode">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="tab"
                  aria-selected={mode === m.id}
                  className={mode === m.id ? 'is-active' : ''}
                  onClick={() => setMode(m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="observatory-presentation-toggle"
              onClick={() => setPresentation(true)}
              title="Hide controls for a clean recording (Esc to exit)"
            >
              Presentation
            </button>
          </div>
        </header>
      )}

      {!presentation && (
        <DetailsPanel
          store={store}
          collapsed={detailsCollapsed}
          onToggle={() => setDetailsCollapsed((v) => !v)}
        />
      )}

      {presentation && (
        <div className="observatory-presentation-hint">Esc to exit</div>
      )}
    </div>
  )
}
