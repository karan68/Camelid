import { existsSync } from 'node:fs'
import { mkdir, readFile, rm } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import puppeteer from 'puppeteer-core'

const executablePath = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].find(existsSync)
if (!executablePath) throw new Error('Chrome or Edge is required')

const root = resolve(process.env.CAMELID_WORKSPACE_UI_ROOT || '../target/workspace-stage5-qwen3-4b-q4km/ui-workspace')
const out = resolve(process.env.CAMELID_WORKSPACE_UI_OUT || '../target/workspace-stage5-qwen3-4b-q4km')
const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const apiBase = process.env.CAMELID_API || 'http://127.0.0.1:8186'
await rm(root, { recursive: true, force: true })
await mkdir(root, { recursive: true })

const browser = await puppeteer.launch({ executablePath, headless: 'new' })
try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1280, height: 800 })
  await page.evaluateOnNewDocument((api, workspace) => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.setItem('camelid.apiBase', api)
    localStorage.setItem('camelid.activeTab', 'workspace')
    localStorage.setItem('camelid.selectedModelId', 'qwen3_4b_q4_k_m')
    localStorage.setItem('camelid.workspacePath', workspace)
  }, apiBase, root)
  await page.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await page.waitForSelector('.workspace-view')
  await page.waitForFunction(() => !document.querySelector('.workspace-blocked'), { timeout: 15000 })
  await page.type('.workspace-field--goal textarea', 'Create a file named ui-greeting.txt whose exact contents are: hello there\nUse the write_file tool ONCE, then reply in words that you created it. Do not call any further tools and do not read the file back.')
  const start = await page.waitForSelector('.workspace-setup__actions .cx-btn--primary:not([disabled])')
  await start.click()
  await page.waitForSelector('.workspace-approval', { timeout: 120000 })
  const approval = await page.$eval('.workspace-approval pre', (node) => node.innerText)
  if (!approval.includes('write_file → ui-greeting.txt') || !approval.includes('--- proposed content ---\nhello there')) {
    throw new Error(`approval did not disclose the exact expected write: ${approval}`)
  }
  await page.screenshot({ path: join(out, 'real-ui-approval.png') })
  const allow = await page.$('.workspace-approval__actions .cx-btn--primary')
  if (!allow) throw new Error('Allow once button not found')
  await allow.click()
  await page.waitForFunction(() => document.body.innerText.includes('Session finished'), { timeout: 120000 })
  const content = await readFile(join(root, 'ui-greeting.txt'), 'utf8')
  if (content !== 'hello there') throw new Error(`UI-approved file mismatch: ${JSON.stringify(content)}`)
  await page.screenshot({ path: join(out, 'real-ui-terminal.png') })
  const metrics = await page.evaluate(() => ({
    document_width: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
    phase: document.querySelector('.workspace-status')?.innerText,
    timeline_rows: document.querySelectorAll('.workspace-event').length,
  }))
  if (metrics.document_width[0] !== metrics.document_width[1]) throw new Error(`page overflow: ${metrics.document_width}`)
  console.log(JSON.stringify({ approval: 'pass_exact_visible_action', file: 'pass_exact_content', ...metrics }, null, 2))
} finally {
  await browser.close()
}
