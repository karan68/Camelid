/* Design-evidence screenshot harness (Phase 1+).
   Captures every view at the spec's widths in both themes against a running
   dev server. Usage:
     node scripts/capture-views.mjs --out design-evidence/phase-1 [--url http://127.0.0.1:4175] [--themes dark,light] [--widths 1440,390]
   Requires Google Chrome installed (driven via puppeteer-core; no bundled
   browser download). */

import { createHash } from 'node:crypto'
import { mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
}

const baseUrl = args.get('url') || process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outDir = args.get('out') || 'design-evidence/capture'
const themes = (args.get('themes') || 'dark,light').split(',')
const widths = (args.get('widths') || '1440,390').split(',').map(Number)
const onlyViews = args.get('views')?.split(',') || null

const VIEWS = ['chat', 'library', 'api', 'compatibility', 'telemetry', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'observatory']
const HEIGHTS = { 1440: 900, 1024: 768, 768: 1024, 390: 844 }

await mkdir(outDir, { recursive: true })
const captured = []
const browser = await launchBrowser({ purpose: 'the view capture', headless: 'new', args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] })

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
        captured.push(file)
      }
    }
    await page.close()
  }
} finally {
  await browser.close()
}

/* Self-check: two different views captured pixel-identical means the app never
   actually switched views (the hash-routing-doesn't-remount failure mode that
   once produced 40 identical chat screenshots). Fail loudly. */
const hashes = new Map()
for (const file of captured) {
  const digest = createHash('sha256').update(await readFile(file)).digest('hex')
  if (hashes.has(digest)) {
    console.error(`capture self-check FAILED: ${file} is pixel-identical to ${hashes.get(digest)}`)
    process.exit(1)
  }
  hashes.set(digest, file)
}
console.log(`capture self-check passed: ${captured.length} screenshots, all distinct`)
