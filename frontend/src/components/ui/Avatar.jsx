/* Camelid brand sparkle + assistant avatar. Single source of truth
   (previously duplicated in ChatWorkspace, AppSidebar, and TopBar). */

export function Sparkle({ className = '', size = 24, title }) {
  return (
    <svg
      className={`camelid-sparkle-icon ${className}`.trim()}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : 'true'}
      xmlns="http://www.w3.org/2000/svg"
    >
      {title ? <title>{title}</title> : null}
      <path
        d="M12 2.2c.35 4.9 1.6 7.1 4.4 9.9 2.8 2.8 5 4.05 9.9 4.4-4.9.35-7.1 1.6-9.9 4.4-2.8 2.8-4.05 5-4.4 9.9-.35-4.9-1.6-7.1-4.4-9.9-2.8-2.8-5-4.05-9.9-4.4 4.9-.35 7.1-1.6 9.9-4.4C10.4 9.3 11.65 7.1 12 2.2z"
        transform="translate(-2 -4.5) scale(1.16)"
        fill="url(#camelid-sparkle-grad)"
      />
      <defs>
        {/* Instrument gradient: steel → brass → copper (matches --camelid-aurora). */}
        <linearGradient id="camelid-sparkle-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#8fb6dc" />
          <stop offset="52%" stopColor="#b9ad8e" />
          <stop offset="100%" stopColor="#dfa371" />
        </linearGradient>
      </defs>
    </svg>
  )
}

/* Round assistant avatar that frames the sparkle. */
export function Avatar({ size = 30, className = '' }) {
  return (
    <span className={`camelid-avatar ${className}`.trim()} style={{ '--avatar-size': `${size}px` }} aria-hidden="true">
      <Sparkle size={Math.round(size * 0.66)} />
    </span>
  )
}

export default Avatar
