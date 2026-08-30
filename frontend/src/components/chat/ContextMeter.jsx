import { useEffect, useRef, useState } from 'react'
import { IconMemory, IconCheckCircle } from '../ui/icons.jsx'
import {
  composeContextBudget,
  formatTokenCount,
  formatPercent,
} from '../../lib/contextBudget.js'

/* How full the model's context window is, and what is filling it.
 *
 * Collapsed it is a chip in the composer status line; expanded it breaks the
 * window into the segments the prompt actually occupies. Two things here are
 * deliberate and should survive edits:
 *
 *   - The reservation is drawn as its own hatched segment rather than folded
 *     into "used", because it is room set aside for the reply, not spent yet.
 *   - The verified bound is drawn as a marker ON the bar with context beyond it
 *     still rendered as usable. The tested envelope is evidence, not a ceiling;
 *     showing it as a wall is what makes a 40k-token model look like an 8k one.
 *
 * Prompt size is a client estimate and is labelled as one everywhere it shows.
 */
export function ContextMeter({
  contextLength,
  promptTokens,
  systemTokens = 0,
  imageTokens = 0,
  reservedTokens,
  verifiedBound = null,
  executionLane = '',
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    function onPointerDown(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false)
    }
    function onKeyDown(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const budget = composeContextBudget({
    contextLength,
    promptTokens,
    systemTokens,
    imageTokens,
    reservedTokens,
    verifiedBound,
  })

  /* A model whose window we cannot read gets no meter at all rather than a
     guessed one — an invented denominator is worse than no denominator. */
  if (!budget) return null

  const summary = `${formatTokenCount(budget.usedTokens + budget.reservedTokens)} / ${formatTokenCount(budget.contextLength)} tokens`

  return (
    <div className="ctxmeter" ref={rootRef}>
      <button
        type="button"
        className={`ctxmeter__chip is-${budget.level}`}
        aria-expanded={open}
        aria-label={`Context window ${formatPercent(budget.filledPercent)} used. ${summary}. Show breakdown.`}
        onClick={() => setOpen((value) => !value)}
      >
        <IconMemory size={13} />
        <span className="ctxmeter__track" aria-hidden="true">
          <span className="ctxmeter__fill" style={{ width: `${Math.min(budget.usedPercent, 100)}%` }} />
          <span
            className="ctxmeter__fill ctxmeter__fill--reserved"
            style={{ left: `${Math.min(budget.usedPercent, 100)}%`, width: `${budget.reservedPercent}%` }}
          />
        </span>
        <span className="ctxmeter__chip-value">{formatPercent(budget.filledPercent)}</span>
      </button>

      {open && (
        <div className="ctxmeter__panel" role="group" aria-label="Context window breakdown">
          <div className="ctxmeter__panel-head">
            <span className="ctxmeter__panel-title">Context window</span>
            <span className="ctxmeter__panel-total">{formatPercent(budget.filledPercent)}</span>
          </div>
          <p className="ctxmeter__panel-count">
            {budget.usedTokens.toLocaleString()} + {budget.reservedTokens.toLocaleString()} reserved
            {' / '}
            {budget.contextLength.toLocaleString()} tokens
          </p>

          <div className="ctxmeter__bar">
            <span className="ctxmeter__bar-fill" style={{ width: `${Math.min(budget.usedPercent, 100)}%` }} />
            <span
              className="ctxmeter__bar-fill ctxmeter__bar-fill--reserved"
              style={{ left: `${Math.min(budget.usedPercent, 100)}%`, width: `${budget.reservedPercent}%` }}
            />
            {budget.showVerifiedMarker && (
              <span
                className="ctxmeter__marker"
                style={{ left: `${budget.verifiedPercent}%` }}
                aria-hidden="true"
              />
            )}
          </div>

          {budget.showVerifiedMarker && (
            <p className="ctxmeter__verified">
              <IconCheckCircle size={12} />
              <span>
                Verified to {budget.verifiedBound.toLocaleString()} tokens.
                {' '}
                {budget.beyondVerified
                  ? 'You are past the tested envelope — still supported, just untested.'
                  : 'Context beyond the marker works; it simply has no committed evidence yet.'}
              </span>
            </p>
          )}

          <ul className="ctxmeter__legend">
            {budget.segments.map((segment) => (
              <li key={segment.key} className={`ctxmeter__legend-row is-${segment.key}`}>
                <span className="ctxmeter__swatch" aria-hidden="true" />
                <span className="ctxmeter__legend-label">{segment.label}</span>
                <span className="ctxmeter__legend-tokens">{segment.tokens.toLocaleString()}</span>
                <span className="ctxmeter__legend-percent">{formatPercent(segment.percent)}</span>
              </li>
            ))}
          </ul>

          <p className="ctxmeter__foot">
            {executionLane
              ? <>Runs on <code>{executionLane}</code> — your hardware, no per-token cost.</>
              : <>Runs on your hardware — no per-token cost.</>}
          </p>
          <p className="ctxmeter__foot ctxmeter__foot--estimate">
            Prompt size is an estimate until the message is sent.
          </p>
        </div>
      )}
    </div>
  )
}

export default ContextMeter
