import { useId, useRef, useState } from 'react'
import { ToolsPopover } from '../mcp/ToolsPopover'
import { IconClose, IconChevronDown } from '../ui/icons'

export function ComposerMenu({ label, icon, children, description, disabled = false }) {
  const [open, setOpen] = useState(false)
  const anchorRef = useRef(null)
  const id = useId()
  const close = restore => { setOpen(false); if (restore) anchorRef.current?.focus({ preventScroll: true }) }
  return <>
    <button ref={anchorRef} type="button" className={'cxcomposer__tool composer-menu-trigger' + (open ? ' is-on' : '')}
      aria-label={label} title={description} aria-expanded={open} aria-controls={open ? id : undefined} aria-haspopup="dialog" disabled={disabled} onClick={() => setOpen(!open)}>
      {icon}<span>{label}</span><IconChevronDown size={12} />
    </button>
    {open && <ToolsPopover anchorRef={anchorRef} id={id} titleId={id + '-title'} onClose={close}
      className="composer-menu" width={440} initialFocus=".composer-menu__close">
      <header className="mcp-picker__head"><div><h2 id={id + '-title'}>{label === 'Options' ? 'Message options' : 'Attach to this message'}</h2>
        <p>{label === 'Options' ? 'Choose how Camelid handles your next message.' : (description || 'Add a document or an image.')}</p></div>
        <button type="button" className="mcp-picker__close composer-menu__close" aria-label={'Close ' + label.toLowerCase()} onClick={() => close(true)}><IconClose size={16} /></button>
      </header>
      <div className="composer-menu__content">{typeof children === 'function' ? children(() => close(true)) : children}</div>
    </ToolsPopover>}
  </>
}
