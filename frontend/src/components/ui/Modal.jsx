import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { IconClose } from './icons'
import { IconButton } from './IconButton'

const FOCUSABLE_SELECTOR = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/* Keeps Tab/Shift+Tab cycling inside `panel`. Shared with the few surfaces
   (ShortcutsOverlay, ModelInspector) that render their own dialog chrome. */
export function wrapDialogFocus(event, panel) {
  if (event.key !== 'Tab' || !panel) return
  const focusables = panel.querySelectorAll(FOCUSABLE_SELECTOR)
  if (!focusables.length) {
    event.preventDefault()
    panel.focus()
    return
  }
  const first = focusables[0]
  const last = focusables[focusables.length - 1]
  const active = document.activeElement
  if (event.shiftKey) {
    if (active === first || active === panel || !panel.contains(active)) {
      event.preventDefault()
      last.focus()
    }
  } else if (active === last || !panel.contains(active)) {
    event.preventDefault()
    first.focus()
  }
}

/* Modal — accessible dialog with backdrop, Esc-to-close, scroll lock, focus capture. */
export function Modal({ open, onClose, title, children, footer = null, labelledById, size = 'md', className = '' }) {
  const panelRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const previouslyFocused = typeof document !== 'undefined' ? document.activeElement : null
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose?.()
        return
      }
      wrapDialogFocus(event, panelRef.current)
    }
    window.addEventListener('keydown', onKey)
    const { body } = document
    const prevOverflow = body.style.overflow
    body.style.overflow = 'hidden'
    const frame = window.requestAnimationFrame(() => panelRef.current?.focus())
    return () => {
      window.removeEventListener('keydown', onKey)
      body.style.overflow = prevOverflow
      window.cancelAnimationFrame(frame)
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') previouslyFocused.focus()
    }
  }, [open, onClose])

  if (!open || typeof document === 'undefined') return null

  return createPortal(
    <div className="cx-modal-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      <div
        ref={panelRef}
        className={`cx-modal cx-modal--${size} ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledById}
        tabIndex={-1}
      >
        {title && (
          <header className="cx-modal__header">
            <h2 id={labelledById} className="cx-modal__title">{title}</h2>
            <IconButton label="Close" size="sm" onClick={onClose}><IconClose size={18} /></IconButton>
          </header>
        )}
        <div className="cx-modal__body">{children}</div>
        {footer && <footer className="cx-modal__footer">{footer}</footer>}
      </div>
    </div>,
    document.body,
  )
}

export default Modal
