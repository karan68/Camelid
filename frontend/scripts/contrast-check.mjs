/* WCAG AA contrast audit for the Camelid token palettes (Phase 1 gate).
   Parses src/styles/tokens.css, resolves the dark and light palettes (the
   light-dark(<light>, <dark>) form each token carries), alpha-composites
   rgba() values over the theme background, and asserts AA for every
   text-on-surface and status-color-on-surface pair the UI actually renders —
   including the computed color-mix() chip/button fills from ui.css.

   Normal text needs 4.5:1. Tokens used only at large/bold sizes or as
   non-text indicators (dots, borders) are checked at 3:1 and marked so. */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const css = readFileSync(join(here, '..', 'src', 'styles', 'tokens.css'), 'utf8')

function extractBlock(source, opener) {
  const start = source.indexOf(opener)
  if (start === -1) throw new Error(`block not found: ${opener}`)
  let depth = 0
  let i = source.indexOf('{', start)
  const open = i
  for (; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1
    if (source[i] === '}') {
      depth -= 1
      if (depth === 0) break
    }
  }
  return source.slice(open + 1, i)
}

function parseVars(block) {
  const vars = {}
  for (const match of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    vars[match[1]] = match[2].trim()
  }
  return vars
}

function parseColor(value) {
  let m = value.match(/^#([0-9a-f]{6})$/i)
  if (m) {
    const n = parseInt(m[1], 16)
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: 1 }
  }
  m = value.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$/i)
  if (m) return { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] }
  if (value === 'transparent') return { r: 0, g: 0, b: 0, a: 0 }
  return null
}

/* Split on commas that sit outside any parentheses (function arguments). */
function splitTopLevel(s) {
  const parts = []
  let depth = 0
  let current = ''
  for (const ch of s) {
    if (ch === '(') depth += 1
    if (ch === ')') depth -= 1
    if (ch === ',' && depth === 0) {
      parts.push(current.trim())
      current = ''
    } else {
      current += ch
    }
  }
  if (current.trim()) parts.push(current.trim())
  return parts
}

/* Resolve a CSS color expression to rgba for one theme. Handles the forms the
   token sheet and ui.css actually use: #rrggbb, rgb()/rgba(), transparent,
   var(--token), light-dark(<light>, <dark>), and
   color-mix(in srgb, <color> [p%], <color> [q%]) (trailing-percent syntax). */
function resolveColorValue(value, theme, lookup) {
  const v = value.trim()
  let m = v.match(/^var\((--[\w-]+)\)$/)
  if (m) return resolveColorValue(lookup(m[1]), theme, lookup)
  m = v.match(/^light-dark\((.+)\)$/is)
  if (m) {
    const args = splitTopLevel(m[1])
    if (args.length !== 2) return null
    return resolveColorValue(theme === 'light' ? args[0] : args[1], theme, lookup)
  }
  m = v.match(/^color-mix\(\s*in\s+srgb\s*,(.+)\)$/is)
  if (m) {
    const args = splitTopLevel(m[1])
    if (args.length !== 2) return null
    const components = args.map((arg) => {
      const pm = arg.match(/^(.*?)\s+([\d.]+)%$/s)
      const color = resolveColorValue(pm ? pm[1] : arg, theme, lookup)
      return color ? { color, weight: pm ? +pm[2] : null } : null
    })
    if (components.some((c) => c === null)) return null
    let [c1, c2] = components
    if (c1.weight === null && c2.weight === null) { c1.weight = 50; c2.weight = 50 }
    else if (c1.weight === null) c1.weight = 100 - c2.weight
    else if (c2.weight === null) c2.weight = 100 - c1.weight
    const total = c1.weight + c2.weight
    const w1 = c1.weight / total
    const w2 = c2.weight / total
    const a = c1.color.a * w1 + c2.color.a * w2
    if (a === 0) return { r: 0, g: 0, b: 0, a: 0 }
    return {
      r: (c1.color.r * c1.color.a * w1 + c2.color.r * c2.color.a * w2) / a,
      g: (c1.color.g * c1.color.a * w1 + c2.color.g * c2.color.a * w2) / a,
      b: (c1.color.b * c1.color.a * w1 + c2.color.b * c2.color.a * w2) / a,
      a,
    }
  }
  return parseColor(v)
}

function composite(fg, bg) {
  const a = fg.a
  return {
    r: fg.r * a + bg.r * (1 - a),
    g: fg.g * a + bg.g * (1 - a),
    b: fg.b * a + bg.b * (1 - a),
    a: 1,
  }
}

function luminance({ r, g, b }) {
  const lin = (c) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

function contrast(fg, bg) {
  const l1 = luminance(fg)
  const l2 = luminance(bg)
  const [hi, lo] = l1 >= l2 ? [l1, l2] : [l2, l1]
  return (hi + 0.05) / (lo + 0.05)
}

const darkVars = parseVars(extractBlock(css, ':root {'))
const lightVars = parseVars(extractBlock(css, ":root[data-theme='light']"))

/* Text + status tokens, checked against every surface they can sit on.
   level: 'AA-normal' (4.5) or 'AA-large' (3.0, also non-text indicators). */
const CHECKS = [
  { fg: '--color-text', level: 'AA-normal' },
  { fg: '--color-text-muted', level: 'AA-normal' },
  { fg: '--color-text-faint', level: 'AA-normal' },
  { fg: '--color-accent-text', level: 'AA-normal' },
  { fg: '--color-verified', level: 'AA-normal' },
  { fg: '--color-evidence', level: 'AA-normal' },
  { fg: '--color-planned', level: 'AA-normal' },
  { fg: '--color-unsupported', level: 'AA-normal' },
  { fg: '--color-ready', level: 'AA-normal' },
  { fg: '--color-warning', level: 'AA-normal' },
  { fg: '--color-error', level: 'AA-normal' },
  { fg: '--color-info', level: 'AA-normal' },
]
const SURFACES = [
  '--color-bg',
  '--color-bg-elevated',
  '--color-surface',
  '--color-surface-subtle',
  '--color-surface-strong',
  '--color-surface-hover',
  '--color-user-chip',
  '--color-code-bg',
]

/* Fill pairs: text drawn on a colored fill. */
const FILL_PAIRS = [
  { fg: '--color-accent-ink', bg: '--color-accent', level: 'AA-normal' },
  { fg: '--color-verified-ink', bg: '--color-verified', level: 'AA-normal' },
  /* .cx-btn--danger:hover (ui.css) */
  { fg: '--color-error-ink', bg: '--color-error', level: 'AA-normal' },
]

/* Computed component fills: state-colored text on the translucent color-mix()
   tints ui.css actually renders (.cx-chip--*, .cx-btn--danger, .cx-btn--tonal
   hover). Each is checked over every surface the chips/buttons sit on; keep
   the recipes in sync with src/styles/ui.css. */
const TINTED_PAIRS = [
  { name: '.cx-chip--ready', fg: '--color-ready', fill: 'color-mix(in srgb, var(--color-ready) 16%, transparent)', level: 'AA-normal' },
  { name: '.cx-chip--warn', fg: '--color-warning', fill: 'color-mix(in srgb, var(--color-warning) 18%, transparent)', level: 'AA-normal' },
  { name: '.cx-chip--error', fg: '--color-error', fill: 'color-mix(in srgb, var(--color-error) 16%, transparent)', level: 'AA-normal' },
  { name: '.cx-chip--info', fg: '--color-info', fill: 'color-mix(in srgb, var(--color-info) 16%, transparent)', level: 'AA-normal' },
  { name: '.cx-btn--danger', fg: '--color-error', fill: 'color-mix(in srgb, var(--color-error) 14%, transparent)', level: 'AA-normal' },
  { name: '.cx-btn--tonal:hover', fg: '--color-accent-text', fill: 'color-mix(in srgb, var(--color-accent) 18%, transparent)', level: 'AA-normal' },
]
const TINT_SURFACES = [
  '--color-bg',
  '--color-bg-elevated',
  '--color-surface',
  '--color-surface-subtle',
  '--color-surface-hover',
]

/* Real, pre-existing AA misses kept visible without turning the gate red.
   Each maps "theme <pair label>" to a floor just under the ratio it achieves
   today, so any further regression still fails. Empty as of 2026-08-05 —
   the light ready/warning/error inks and dark --color-text-faint were nudged
   in tokens.css and every pair now clears AA. Add entries here only for a
   genuine, documented exception. */
const KNOWN_BELOW_AA = new Map([])

const MIN = { 'AA-normal': 4.5, 'AA-large': 3.0 }

let failures = 0
let known = 0
let checks = 0

function judge(themeName, label, ratio, level) {
  checks += 1
  const min = MIN[level]
  const key = `${themeName} ${label}`
  const floor = KNOWN_BELOW_AA.get(key)
  if (ratio >= min) {
    if (floor !== undefined) {
      console.warn(`NOTE [${themeName}] ${label} now passes (${ratio.toFixed(2)}) — remove its KNOWN_BELOW_AA entry`)
    }
    return
  }
  if (floor !== undefined && ratio >= floor) {
    known += 1
    console.warn(`KNOWN [${themeName}] ${label}: ${ratio.toFixed(2)} < ${min} (${level}) — pre-existing; needs a token nudge`)
    return
  }
  failures += 1
  console.error(`FAIL [${themeName}] ${label}: ${ratio.toFixed(2)} < ${min} (${level})`)
}

for (const themeName of ['dark', 'light']) {
  const lookup = (name) => {
    const raw = (themeName === 'light' ? lightVars[name] : undefined) ?? darkVars[name]
    if (!raw) throw new Error(`missing token ${name} in ${themeName}`)
    return raw
  }
  const resolve = (name) => {
    const color = resolveColorValue(lookup(name), themeName, lookup)
    if (!color) throw new Error(`unparseable color ${name}: ${lookup(name)} (${themeName})`)
    return color
  }
  const themeBg = resolve('--color-bg')
  const surface = (name) => composite(resolve(name), themeBg)

  for (const check of CHECKS) {
    for (const surfaceName of SURFACES) {
      const fg = composite(resolve(check.fg), surface(surfaceName))
      const ratio = contrast(fg, surface(surfaceName))
      judge(themeName, `${check.fg} on ${surfaceName}`, ratio, check.level)
    }
  }
  for (const pair of FILL_PAIRS) {
    const fill = composite(resolve(pair.bg), themeBg)
    const fg = composite(resolve(pair.fg), fill)
    judge(themeName, `${pair.fg} on ${pair.bg}`, contrast(fg, fill), pair.level)
  }
  for (const pair of TINTED_PAIRS) {
    const tint = resolveColorValue(pair.fill, themeName, lookup)
    if (!tint) throw new Error(`unparseable fill for ${pair.name}: ${pair.fill} (${themeName})`)
    for (const surfaceName of TINT_SURFACES) {
      const fill = composite(tint, surface(surfaceName))
      const fg = composite(resolve(pair.fg), fill)
      judge(themeName, `${pair.name} (${pair.fg}) on ${surfaceName}`, contrast(fg, fill), pair.level)
    }
  }
}

if (failures > 0) {
  console.error(`\ncontrast-check: ${failures}/${checks} pairs below WCAG AA`)
  process.exit(1)
}
if (known > 0) {
  console.log(`contrast-check: ${checks - known}/${checks} token + component-fill pairs pass WCAG AA in both themes; ${known} known pre-existing miss${known === 1 ? '' : 'es'} (listed above) held at their current ratios`)
} else {
  console.log(`contrast-check: all ${checks} token + component-fill pairs pass WCAG AA in both themes`)
}
