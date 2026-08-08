import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'camelid-theme'
const VALID = new Set(['system', 'light', 'dark'])
/* Single source of truth for the toggle cycle; ThemeToggle derives its
   "Switch to …" label from this so the two can never drift. */
export const THEME_ORDER = ['dark', 'light', 'system']

/* Dark is the design's canonical palette, so it is the default preference;
   'system' and 'light' remain one toggle away. */
function readPreference() {
  if (typeof window === 'undefined') return 'dark'
  const saved = window.localStorage.getItem(STORAGE_KEY)
  return saved && VALID.has(saved) ? saved : 'dark'
}

function systemPrefersDark() {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

/* Browser/OS chrome color per theme; mirrors --color-bg in styles/tokens.css. */
const THEME_COLOR = { light: '#f6f8fa', dark: '#0e1216' }

/* Keep index.html's media-scoped theme-color metas in step with the resolved
   theme: an explicit preference pins both to that palette's canvas; 'system'
   restores each tag's own per-scheme default so the OS preference drives it. */
function syncThemeColorMeta(preference) {
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    if (preference === 'system') {
      const media = meta.getAttribute('media') || ''
      meta.setAttribute('content', media.includes('light') ? THEME_COLOR.light : THEME_COLOR.dark)
    } else {
      meta.setAttribute('content', THEME_COLOR[preference])
    }
  }
}

function applyPreference(preference) {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  if (preference === 'system') {
    // Remove the attribute so the prefers-color-scheme media query drives the palette.
    delete root.dataset.theme
  } else {
    root.dataset.theme = preference
  }
  syncThemeColorMeta(preference)
}

/**
 * Dual light/dark theme, system-following.
 *  - preference ∈ { 'system', 'light', 'dark' } (persisted)
 *  - 'system' removes [data-theme] so CSS prefers-color-scheme wins, and a live
 *    matchMedia listener keeps `resolved` in sync for the toggle UI.
 *  - 'light' / 'dark' set [data-theme] explicitly.
 */
export function useTheme() {
  const [preference, setPreferenceState] = useState(readPreference)
  const [resolved, setResolved] = useState(() =>
    preference === 'system' ? (systemPrefersDark() ? 'dark' : 'light') : preference,
  )

  useEffect(() => {
    applyPreference(preference)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, preference)
    }
    if (preference !== 'system') {
      setResolved(preference)
      return undefined
    }
    if (typeof window === 'undefined' || !window.matchMedia) {
      setResolved('light')
      return undefined
    }
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const sync = () => setResolved(media.matches ? 'dark' : 'light')
    sync()
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [preference])

  const setPreference = useCallback((next) => {
    setPreferenceState(VALID.has(next) ? next : 'system')
  }, [])

  const cyclePreference = useCallback(() => {
    setPreferenceState((current) => {
      const index = THEME_ORDER.indexOf(current)
      return THEME_ORDER[(index + 1) % THEME_ORDER.length]
    })
  }, [])

  return { preference, setPreference, cyclePreference, resolved }
}
