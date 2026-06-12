/* Design-evidence screenshot harness (Phase 1+).
   Captures every view at the spec's widths in both themes against a running
   dev server. Usage:
     node scripts/capture-views.mjs --out design-evidence/phase-1 [--url http://127.0.0.1:4175] [--themes dark,light] [--widths 1440,390]
   Requires Google Chrome installed (driven via puppeteer-core; no bundled
   browser download). */

import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import puppeteer from 'puppeteer-core'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
}

const baseUrl = args.get('url') || process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outDir = args.get('out') || 'design-evidence/capture'
const themes = (args.get('themes') || 'dark,light').split(',')
const widths = (args.get('widths') || '1440,390').split(',').map(Number)
const onlyViews = args.get('views')?.split(',') || null

const VIEWS = ['chat', 'library', 'api', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'observatory']
const HEIGHTS = { 1440: 900, 1024: 768, 768: 1024, 390: 844 }

const CHROME_PATHS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
]
const { existsSync } = await import('node:fs')
const executablePath = CHROME_PATHS.find((p) => existsSync(p))
if (!executablePath) throw new Error('No local Chrome found for capture')

await mkdir(outDir, { recursive: true })
const browser = await puppeteer.launch({ executablePath, headless: 'new' })

try {
  for (const theme of themes) {
    const page = await browser.newPage()
    // Seed the persisted theme preference before the app boots.
    await page.evaluateOnNewDocument((t) => {
      window.localStorage.setItem('camelid-theme', t)
    }, theme)
    for (const width of widths) {
      await page.setViewport({ width, height: HEIGHTS[width] || 900 })
      for (const view of (onlyViews || VIEWS)) {
        // The app reads the hash once on mount, so hash-only navigation would
        // not switch tabs — force a fresh document load for every view.
        await page.goto('about:blank')
        await page.goto(`${baseUrl}/#${view}`, { waitUntil: 'networkidle2', timeout: 30000 })
        await new Promise((resolve) => setTimeout(resolve, 900))
        const file = join(outDir, `${view}-${theme}-${width}.png`)
        await page.screenshot({ path: file })
        console.log(`captured ${file}`)
      }
    }
    await page.close()
  }
} finally {
  await browser.close()
}
