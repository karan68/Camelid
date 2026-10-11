// Live check for #801: the real chat UI, searching the whole 550-PDF library.
// node live801.mjs <ui origin> <label> <out dir>
// Turns on "Whole library" from the composer's Attach menu, asks two planted
// questions in new chats, and records what the chat sent and answered.
import { writeFileSync, mkdirSync } from 'node:fs'
import { launchBrowser } from './scripts/lib/launch-browser.mjs'

const [origin, label, outDir] = process.argv.slice(2)
mkdirSync(outDir, { recursive: true })
const QUESTIONS = [
  [21, 'How much did it cost when the primary mirror of the Ivarsdal observatory was recoated?'],
  [13, 'How much did it cost when the north clarifier of the Wyncroft water treatment plant was relined?'],
]
const browser = await launchBrowser({ purpose: 'the #801 live check', headless: 'new', protocolTimeout: 900000 })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
const searches = []
page.on('request', (request) => {
  if (request.url().includes('/api/documents/search') && request.method() === 'POST') searches.push({ body: JSON.parse(request.postData()) })
})
page.on('response', async (response) => {
  if (!response.url().includes('/api/documents/search')) return
  const body = await response.json().catch(() => null)
  const entry = searches.findLast((s) => !s.response)
  if (entry) entry.response = { status: response.status(), mode: body?.retrieval?.mode, results: (body?.results || []).map((r) => ({ filename: r.filename, excerpt: r.excerpt?.slice(0, 160) })) }
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('live801')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('live801', '1')
})
const composer = 'textarea[aria-label="Message Camelid"]:not([disabled])'
await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
await page.waitForSelector(composer, { timeout: 60000 })
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const results = []
for (const [id, question] of QUESTIONS) {
  if (results.length) {
    const clicked = await page.$$eval('button', (nodes) => {
      const node = nodes.find((n) => n.textContent.trim() === 'New chat' && n.offsetParent !== null)
      node?.click()
      return Boolean(node)
    })
    if (!clicked) throw new Error('no visible New chat button')
    await page.waitForFunction(() => document.querySelectorAll('.cxturn').length === 0, { timeout: 10000 })
  }
  await page.waitForSelector(composer, { timeout: 30000 })
  if (!(await page.$('button[aria-label="Stop searching the whole library"]'))) {
    await page.click('button[aria-label="Attach"]')
    await page.waitForSelector('button[aria-label="Search the whole library"]', { timeout: 5000 })
    await page.click('button[aria-label="Search the whole library"]')
    await page.waitForSelector('button[aria-label="Stop searching the whole library"]', { timeout: 5000 })
  }
  await sleep(3000)
  const pill = await page.$eval('.cxcomposer__doc-pill--library', (node) => node.innerText).catch(() => null)
  await page.$eval(composer, (node, next) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(node, next)
    node.dispatchEvent(new Event('input', { bubbles: true }))
  }, question)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
  await page.screenshot({ path: `${outDir}/${label}-q${id}-1-asking.png` })
  const before = searches.length
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction(() => document.querySelectorAll('.cxturn--assistant').length > 0, { timeout: 120000 })
  await sleep(1500)
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop'), { timeout: 600000, polling: 500 })
  await sleep(1500)
  const reply = await page.$$eval('.cxturn--assistant', (nodes) => nodes.at(-1).innerText)
  await page.screenshot({ path: `${outDir}/${label}-q${id}-2-reply.png` })
  results.push({ id, question, library_pill: pill, searches: searches.slice(before), reply })
  console.log(`${label} q${id}: pill=${JSON.stringify(pill)} searches=${searches.length - before} reply=${reply.replace(/\s+/g, ' ').slice(0, 240)}`)
}
writeFileSync(`${outDir}/${label}.json`, JSON.stringify({ ui: label, results }, null, 1))
await browser.close()
