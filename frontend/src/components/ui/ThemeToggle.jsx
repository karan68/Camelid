import { THEME_ORDER } from '../../hooks/useTheme'
import { IconMonitor, IconMoon, IconSun } from './icons'

/* Cycles theme preference dark → light → system (THEME_ORDER). Shows the active preference. */
export function ThemeToggle({ preference, resolved, onCycle, compact = false }) {
  const Icon = preference === 'system' ? IconMonitor : preference === 'light' ? IconSun : IconMoon
  const labelFor = { system: 'System', light: 'Light', dark: 'Dark' }
  const next = THEME_ORDER[(THEME_ORDER.indexOf(preference) + 1) % THEME_ORDER.length]
  const aria = `Theme: ${labelFor[preference]} (${resolved}). Switch to ${labelFor[next]}.`
  return (
    <button
      type="button"
      className={`cx-theme-toggle ${compact ? 'is-compact' : ''}`.trim()}
      onClick={onCycle}
      aria-label={aria}
      title={aria}
    >
      <span className="cx-theme-toggle__icon"><Icon size={18} /></span>
      {!compact && <span className="cx-theme-toggle__label">{labelFor[preference]}</span>}
    </button>
  )
}

export default ThemeToggle
