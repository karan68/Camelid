import { useEffect, useState } from 'react'

/* Tooltip — lightweight CSS hover/focus tooltip. Wraps a single trigger.
   Hidden from assistive tech until triggered; Escape dismisses (WCAG 1.4.13).
   placement: top | bottom | right | left */
export function Tooltip({ content, placement = 'top', className = '', children }) {
  const [active, setActive] = useState(false)
  const [dismissed, setDismissed] = useState(false)
  const shown = active && !dismissed

  useEffect(() => {
    if (!shown) return undefined
    const onKeyDown = (event) => { if (event.key === 'Escape') setDismissed(true) }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [shown])

  if (!content) return children

  const show = () => setActive(true)
  const hide = () => { setActive(false); setDismissed(false) }

  return (
    <span
      className={`cx-tooltip cx-tooltip--${placement} ${className}`.trim()}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      <span
        role="tooltip"
        aria-hidden={shown ? undefined : 'true'}
        className="cx-tooltip__bubble"
        style={dismissed ? { opacity: 0, visibility: 'hidden' } : undefined}
      >
        {content}
      </span>
    </span>
  )
}

export default Tooltip
