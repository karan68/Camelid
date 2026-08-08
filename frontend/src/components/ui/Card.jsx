/* Card — elevated surface. Optional title/eyebrow/actions header. tone: default | accent | muted.
   interactive + onClick gets button semantics (role, tabIndex, Enter/Space) built in;
   a caller-supplied onKeyDown takes over keyboard handling entirely. */
export function Card({ tone = 'default', interactive = false, className = '', children, as: Tag = 'section', onClick, onKeyDown, ...rest }) {
  const classes = [
    'cx-card',
    `cx-card--${tone}`,
    interactive ? 'cx-card--interactive' : '',
    className,
  ].filter(Boolean).join(' ')

  const clickable = interactive && typeof onClick === 'function'
  const activateOnKeyDown = (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    onClick(event)
  }

  return (
    <Tag
      className={classes}
      onClick={onClick}
      onKeyDown={onKeyDown || (clickable ? activateOnKeyDown : undefined)}
      {...(clickable ? { role: 'button', tabIndex: 0 } : null)}
      {...rest}
    >
      {children}
    </Tag>
  )
}

export function CardHeader({ eyebrow, title, icon = null, actions = null, className = '' }) {
  return (
    <header className={`cx-card__header ${className}`.trim()}>
      {icon && <span className="cx-card__header-icon">{icon}</span>}
      <div className="cx-card__header-copy">
        {eyebrow && <span className="cx-card__eyebrow">{eyebrow}</span>}
        {title && <h3 className="cx-card__title">{title}</h3>}
      </div>
      {actions && <div className="cx-card__header-actions">{actions}</div>}
    </header>
  )
}

export function CardBody({ className = '', children }) {
  return <div className={`cx-card__body ${className}`.trim()}>{children}</div>
}

export default Card
