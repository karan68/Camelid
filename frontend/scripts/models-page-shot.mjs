/* One-off full-page screenshot of the Models (#library) view for before/after
   evidence. Windows-friendly (finds Chrome/Edge). Usage:
     node scripts/models-page-shot.mjs --out qa/models-before.png [--url http://127.0.0.1:4175] [--width 1280] */
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
const url = args.get('url') || 'http://127.0.0.1:4175'
const out = args.get('out') || 'models-page.png'
const width = Number(args.get('width') || 1280)

await mkdir(dirname(out), { recursive: true }).catch(() => {})
const browser = await launchBrowser({ purpose: 'the models page shot', headless: 'new' })
try {
  const page = await browser.newPage()
  await page.setViewport({ width, height: 900 })
  await page.goto(`${url}/#library`, { waitUntil: 'networkidle2', timeout: 30000 })
  await new Promise((r) => setTimeout(r, 2500))
  await page.screenshot({ path: out, fullPage: true })
  console.log(`saved ${out}`)
} finally {
  await browser.close()
}
