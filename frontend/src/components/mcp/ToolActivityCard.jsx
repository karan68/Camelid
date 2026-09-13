import { useEffect, useState } from 'react'
import { Button } from '../ui/Button'
import { IconLink, IconChevronDown } from '../ui/icons'
import { OutputMessageContext, ToolOutputGallery } from '../outputs/OutputActions'

const LABELS = { preparing: 'Preparing request', approval: 'Needs approval', executing: 'Running', complete: 'Completed', denied: 'Denied', interrupted: 'Interrupted', cancelled: 'Cancelled', failed: 'Failed', pending: 'Queued', unavailable: 'Not executed' }
export function ToolActivityCard({ call, result: savedResult, live, queued, approval, onDecision, onStop }) {
  const result = savedResult || live?.result
  const phase = result?.mcp?.status || live?.phase || (queued ? 'pending' : 'unavailable')
  const needsApproval = phase === 'approval' && approval?.id === live?.receiptId
  const active = ['preparing', 'approval', 'executing'].includes(phase)
  const [open, setOpen] = useState(active)
  const [now, setNow] = useState(Date.now)
  useEffect(() => { if (phase === 'approval') setOpen(true); else if (['complete', 'denied', 'cancelled', 'interrupted', 'failed'].includes(phase)) setOpen(false) }, [phase])
  useEffect(() => {
    if (phase !== 'executing' || !live?.startedAt) return undefined
    setNow(Date.now())
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [phase, live?.startedAt])
  const measuredDuration = result?.mcp?.duration_ms ?? (phase === 'executing' && live?.startedAt ? Math.max(0, now - live.startedAt) : null)
  const duration = Number.isFinite(measuredDuration) && measuredDuration >= 0 ? measuredDuration : null
  const error = result?.mcp?.is_error
  let args = call.function?.arguments || '{}'
  try { args = JSON.stringify(JSON.parse(args), null, 2) } catch { /* Preserve malformed arguments for inspection. */ }
  return <section className={'tool-activity is-' + phase + (error ? ' is-error' : '')} aria-label="Connected tool activity">
    <button type="button" className="tool-activity__summary" aria-expanded={open} onClick={() => setOpen(!open)}>
      <span className="tool-activity__icon"><IconLink size={17} /></span>
      <span className="tool-activity__name"><strong>{result?.mcp?.tool || live?.tool || call.function?.name || 'Tool request'}</strong><small>{result?.mcp?.connection || live?.connection || 'Connected tool'}</small></span>
      <span className="tool-activity__status" role={active ? 'status' : undefined}>{error && phase === 'complete' ? 'Tool error' : LABELS[phase] || phase}{duration !== null && <small>{duration < 1000 ? Math.round(duration) + ' ms' : (duration / 1000).toFixed(1) + 's'}</small>}</span>
      <IconChevronDown size={14} className={open ? 'is-expanded' : ''} />
    </button>
    {open && <div className={"tool-activity__body" + (needsApproval ? " mcp-approval" : "")}>
      {phase === 'approval' && <p>Review the arguments before they are sent to <strong>{live?.connection}</strong>.</p>}
      <h4>Request</h4><pre>{args}</pre>
      {result && <div className="tool-activity__result"><h4>Result</h4>
        {savedResult && <OutputMessageContext.Provider value={savedResult.id}><ToolOutputGallery content={savedResult.content} /></OutputMessageContext.Provider>}
        <pre>{result.content}</pre>
      </div>}
      {phase === 'unavailable' && <p>No execution result was recorded. This request will not run automatically.</p>}
      {active && <div className="mcp-actions">
        {needsApproval && <><Button variant="primary" onClick={() => onDecision(true)}>Allow once</Button><Button onClick={() => onDecision(false)}>Deny</Button></>}
        <Button variant="ghost" onClick={onStop}>Stop</Button>
      </div>}
    </div>}
  </section>
}
