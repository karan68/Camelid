import { useEffect, useRef, useState } from 'react'
import { subscribeLifecycle } from '../../lib/telemetryLog'
import { createBench, createChoreography, readPalette } from '../../lib/observatory/flowBench'

/* The Flow Bench (Phase 6.1): inference as liquid, every motion a real event.

   - request start  -> prompt-ink droplet at an inlet, drifting with the bench
   - first content  -> the droplet bursts where it stands (TTFT = drift length)
   - progress       -> generation-ink thread; tokens/sec drives the jet velocity
   - end ok         -> inks mix and diffuse to ambient
   - end interrupted-> the thread cuts and curls back (small counter-splat)
   - end error      -> immiscible low-saturation red bloom, re-splatted briefly
                       so it visibly refuses to mix before dissipating
   - idle           -> no injections; the field settles. Stillness is honest.

   The canvas is aria-hidden: everything it encodes is in the instrument rail
   and request log. Sim consumes COUNTS and TIMINGS only, never content. The
   sim event log (ids only) is exposed for the truth-check harness. */

const DPR_CAP = 2
const tint = ([r, g, b], k) => [r * k, g * k, b * k]

/* Truth-check ledger (module-level, app-lifetime — mirrors the bus): every
   lifecycle event the sim subsystem receives, ids and types only. The canvas
   only animates while mounted (missed motion settles honestly), but this log
   must match the metrics' request log one-to-one. */
const simLog = []
subscribeLifecycle((event) => {
  simLog.push({ type: event.type, id: event.id, at: event.at })
  if (simLog.length > 2000) simLog.splice(0, simLog.length - 2000)
})
if (typeof window !== 'undefined') window.__camelidFlowBenchLog = simLog

export function FlowBench({ reducedMotion = false, highlightId = null, onSimEvent = null }) {
  const canvasRef = useRef(null)
  const overlayRef = useRef(null)
  const [rendererKind, setRendererKind] = useState('')

  useEffect(() => {
    const canvas = canvasRef.current
    const overlay = overlayRef.current
    if (!canvas || !overlay) return undefined

    const bench = createBench(canvas)
    if (!bench) return undefined
    setRendererKind(bench.kind)
    const choreography = createChoreography()
    let palette = readPalette()

    const themeObserver = new MutationObserver(() => { palette = readPalette() })
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)
    const fit = () => {
      const rect = canvas.parentElement.getBoundingClientRect()
      bench.resize(Math.round(rect.width * dpr), Math.round(rect.height * dpr))
      overlay.width = Math.round(rect.width * dpr)
      overlay.height = Math.round(rect.height * dpr)
    }
    fit()
    const resizeObserver = new ResizeObserver(fit)
    resizeObserver.observe(canvas.parentElement)

    /* Late join: a request that started while this view was unmounted is still
       a REAL request — render its remaining lifecycle from the inlet onward
       (the user who sends a chat and then opens the Observatory must see it). */
    const ensureRequest = (event) => {
      if (!choreography.active.has(event.id)) {
        const req = choreography.start({ id: event.id, kind: event.kind || 'late' })
        bench.splat(req.x, req.y, tint(palette.prompt, 0.5), 0.0009)
      }
    }
    const unsubscribe = subscribeLifecycle((event) => {
      onSimEvent?.(event)
      if (event.type === 'start') {
        const req = choreography.start(event)
        bench.splat(req.x, req.y, tint(palette.prompt, 0.5), 0.0009)
      } else if (event.type === 'first_content') {
        ensureRequest(event)
        choreography.firstContent(event)
        choreography.firstContent(event)
      } else if (event.type === 'progress') {
        ensureRequest(event)
        choreography.progress(event)
        const req = choreography.active.get(event.id)
        if (req && req.phase === 'drift') req.phase = 'burst'
      } else if (event.type === 'end') {
        choreography.end(event)
      }
    })

    let raf = null
    let last = performance.now()
    let running = !reducedMotion

    const frame = (now) => {
      raf = null
      const dt = Math.min((now - last) / 1000, 0.05)
      last = now
      const t = now / 1000
      const jets = []

      for (const req of choreography.active.values()) {
        if (req.phase === 'drift') {
          req.x = Math.min(req.x + dt * 0.05, 0.92)
          bench.splat(req.x, req.y, tint(palette.prompt, 0.09), 0.0006)
          choreography.trace(req.id, req.x, req.y)
        } else if (req.phase === 'burst') {
          bench.splat(req.x, req.y, tint(palette.prompt, 0.55), 0.004)
          req.phase = 'flow'
        } else if (req.phase === 'flow') {
          const speed = Math.min(req.tokensPerSec / 24, 1.5)
          jets.push({ x: req.x, y: req.y, power: 0.18 + speed * 0.7 })
          const tx = Math.min(req.x + dt * (0.05 + speed * 0.12), 0.95)
          req.x = tx
          bench.splat(tx, req.y, tint(palette.generation, 0.11), 0.0012)
          choreography.trace(req.id, tx, req.y)
        } else if (req.phase === 'cut') {
          bench.splat(req.x, req.y, tint(palette.generation, 0.25), 0.002)
          jets.push({ x: req.x, y: req.y, power: -0.35 }) // curls back
          req.phase = 'settled'
        } else if (req.phase === 'bloom') {
          // immiscible: re-splat the same spot so it holds shape before fading
          if (!req.bloomUntil) req.bloomUntil = performance.now() + 2600
          if (performance.now() < req.bloomUntil) bench.splat(req.x, req.y, tint(palette.error, 0.05), 0.005)
          else req.phase = 'settled'
        } else if (req.phase === 'mixing') {
          bench.splat(req.x, req.y, tint(palette.generation, 0.3), 0.0015)
          req.phase = 'settled'
        }
      }
      choreography.prune()

      bench.step(t, dt, 0.12, jets)
      bench.render()

      // hover-highlight overlay: draw the hovered request's ink thread
      const octx = overlay.getContext('2d')
      octx.clearRect(0, 0, overlay.width, overlay.height)
      if (highlightRef.current) {
        const trace = choreography.traces.get(highlightRef.current)
        if (trace && trace.length > 1) {
          octx.strokeStyle = 'rgba(255,255,255,0.85)'
          octx.lineWidth = 2 * dpr
          octx.setLineDash([6 * dpr, 4 * dpr])
          octx.beginPath()
          trace.forEach((point, index) => {
            const px = point.x * overlay.width
            const py = point.y * overlay.height
            if (index === 0) octx.moveTo(px, py)
            else octx.lineTo(px, py)
          })
          octx.stroke()
        }
      }

      if (running && !document.hidden) raf = window.requestAnimationFrame(frame)
    }

    const start = () => {
      if (!raf && running && !document.hidden) {
        last = performance.now()
        raf = window.requestAnimationFrame(frame)
      }
    }
    const onVisibility = () => {
      if (document.hidden && raf) {
        window.cancelAnimationFrame(raf)
        raf = null
      } else start()
    }
    document.addEventListener('visibilitychange', onVisibility)

    if (reducedMotion) {
      // one static frame of the current field state; no animation
      running = false
      frame(performance.now())
    } else {
      start()
    }

    return () => {
      running = false
      if (raf) window.cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVisibility)
      resizeObserver.disconnect()
      themeObserver.disconnect()
      unsubscribe()
      bench.destroy()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reducedMotion])

  // highlight id is read inside the rAF loop via a ref so hover never restarts the sim
  const highlightRef = useRef(highlightId)
  useEffect(() => { highlightRef.current = highlightId }, [highlightId])

  return (
    <div className="flowbench" data-renderer={rendererKind}>
      <canvas ref={canvasRef} className="flowbench__canvas" aria-hidden="true" />
      <canvas ref={overlayRef} className="flowbench__overlay" aria-hidden="true" />
    </div>
  )
}

export default FlowBench
