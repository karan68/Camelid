// Real UI, isolated demo data: no model execution, personal storage, or external requests.
// Run after npm run build: node scripts/capture-readme.mjs
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdirSync, readFileSync, existsSync, statSync } from 'node:fs'
import { extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { launchBrowser } from './lib/launch-browser.mjs'

const root = fileURLToPath(new URL('../..', import.meta.url))
const dist = resolve(root, 'frontend/dist')
const out = resolve(root, 'docs/assets/readme')
assert.ok(existsSync(resolve(dist, 'index.html')), 'Run npm run build first')
mkdirSync(out, { recursive: true })
const ledger = JSON.parse(readFileSync(resolve(root, 'ledger/camelid-ledger.json'), 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map(row => row.contract) }
const model = 'Qwen3-0.6B-Q8_0.gguf'
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2', '.woff': 'font/woff' }
let surface = 'full'
const server = createServer((req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname
  const json = value => { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)) }
  if (path === '/v1/health') return json({ ok: true, engine: 'camelid', api_surface: surface, version: 'README demo', backend: 'llama', model_family: 'qwen3', loaded_now: true, generation_ready: true, active_model_id: model, active_context_length: 4096, max_prompt_tokens: 4096, max_generation_tokens: 8192 })
  if (path === '/v1/models') return json({ object: 'list', data: [{ id: model, object: 'model', owned_by: 'camelid', meta: { n_ctx_train: 32768, n_params: 600000000, size: 639446688 } }] })
  if (path === '/api/capabilities') return json(capabilities)
  if (path === '/api/models/local') return json({ models_dir: 'models', models: [{ filename: model, size_bytes: 639446688, architecture: 'qwen3', quantization: 'Q8_0', admitted: true, oracle_qualified: true, chat_capable: true, generation_capable: true, context_length: 32768, lane_class: 'supported' }] })
  if (path === '/api/models/current') return json({ id: model, path: `models/${model}`, gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } }, tokenizer: { status: 'available' } })
  if (path === '/api/models/catalog/downloads') return json([])
  if (path.startsWith('/api/') || path.startsWith('/v1/')) { res.writeHead(404, { 'Content-Type': 'application/json' }); return res.end('{"error":"Not part of the README demo"}') }
  const file = resolve(dist, '.' + path)
  if (file.startsWith(dist + sep) && existsSync(file) && statSync(file).isFile()) {
    res.writeHead(200, { 'Content-Type': mime[extname(file)] || 'application/octet-stream' })
    return res.end(readFileSync(file))
  }
  res.writeHead(200, { 'Content-Type': 'text/html' })
  res.end(readFileSync(resolve(dist, 'index.html')))
})
await new Promise(done => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`
const projects = [
  { id: 'field-notes', name: 'Field notes', instructions: 'Keep answers concise. Turn rough notes into practical next steps.', references: [{ id: 'brief', name: 'project-brief.md', content: 'A local-first notebook for ideas, reading notes, and small experiments. Keep the interface simple and accessible.' }] },
  { id: 'weekend-build', name: 'Weekend build', instructions: 'Prefer small, understandable changes. Explain how to verify the result.', references: [{ id: 'requirements', name: 'requirements.md', content: 'Build a reading list with titles, notes, and a completed state.' }] },
  { id: 'learning', name: 'Learning lab', instructions: 'Explain unfamiliar concepts with concrete examples and short exercises.', references: [] },
]
const conversations = [
  { id: 'welcome', title: 'A calmer place for ideas', pinned: true, context: { project_id: 'field-notes' }, messages: [
    { id: 'u1', role: 'user', content: 'Help me turn my scattered notes into a simple weekly routine.' },
    { id: 'a1', role: 'assistant', finish_reason: 'stop', content: 'Start with three small habits. Keep everything in one place, and make the review short enough to repeat.\n\n### Your weekly rhythm\n\n| When | What to do | Time |\n| --- | --- | --- |\n| During the week | Capture ideas in a single inbox | 2 min |\n| Friday | Keep what matters; archive the rest | 10 min |\n| Monday | Choose one idea to explore | 5 min |\n\n**A useful rule:** every note gets either a next step or a home. You do not need to organize everything today.' },
  ] },
  { id: 'checklist', title: 'A checklist I can reuse', context: { project_id: 'field-notes' }, messages: [
    { id: 'u2', role: 'user', content: 'Make a short weekly review checklist I can save as a Markdown file.' },
    { id: 'a2', role: 'assistant', finish_reason: 'stop', content: 'Here is a reusable checklist. Open **Preview** to read it, or download it for your own notes.\n\n```markdown\n# Weekly review\n\nMake room for the next good idea.\n\n## Clear the inbox\n- [ ] Gather this week\'s loose notes\n- [ ] Keep the ideas worth revisiting\n- [ ] Archive what is no longer useful\n\n## Choose a direction\n- [ ] Pick one idea to explore\n- [ ] Write the smallest next step\n- [ ] Set aside 25 minutes to begin\n```' },
  ] },
  { id: 'reading', title: 'Notes from my reading list', context: { project_id: 'learning' }, messages: [] },
].map(c => ({ ...c, created_at: '2026-09-16T09:00:00Z', updated_at: '2026-09-16T09:00:00Z' }))
const browser = await launchBrowser({ purpose: 'README screenshots', headless: 'new' })
const errors = []
try {
  for (const mobile of [false, true]) {
    surface = mobile ? 'lan_chat_only' : 'full'
    const page = await browser.newPage()
    page.on('pageerror', error => errors.push(String(error)))
    await page.setViewport({ width: mobile ? 390 : 1440, height: mobile ? 844 : 960, deviceScaleFactor: 1, isMobile: mobile, hasTouch: mobile })
    await page.setRequestInterception(true)
    page.on('request', request => {
      const url = request.url()
      if (url.startsWith(origin + '/') || url.startsWith('data:') || url.startsWith('blob:')) request.continue()
      else request.abort()
    })
    const chats = structuredClone(conversations)
    if (mobile) chats[0].messages[1].content = 'Keep one inbox and give each note a next step.\n\n- **Capture** ideas as they arrive.\n- **Review** your notes each Friday.\n- **Choose** one idea for Monday.\n\nStart with ten minutes. Keep the habit small.'
    await page.evaluateOnNewDocument((chats, items, base) => {
      if (location.origin !== base) return
      if (sessionStorage.getItem('readme-seeded')) return
      localStorage.setItem('camelid-theme', 'dark')
      localStorage.setItem('camelid.conversations', JSON.stringify(chats))
      localStorage.setItem('camelid.projects', JSON.stringify(items))
      localStorage.setItem('camelid.selectedConversationId', 'welcome')
      localStorage.setItem('camelid.sidebarCollapsed', 'false')
      localStorage.setItem('camelid.webResearchEnabled', 'false')
      sessionStorage.setItem('readme-seeded', 'true')
    }, chats, projects, origin)
    const visit = async (view = 'chat') => {
      await page.goto('about:blank')
      await page.goto(`${origin}/?view=${view}#${view}`, { waitUntil: 'networkidle0' })
      await page.evaluate(() => document.fonts.ready)
    }
    const capture = async name => {
      await new Promise(done => setTimeout(done, 250))
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: horizontal overflow`)
      await page.screenshot({ path: resolve(out, name + '.png') })
      console.log(`Captured ${name}`)
    }
    await visit()
    await page.waitForSelector('[aria-label="Message Camelid"]')
    await capture(mobile ? 'mobile-chat' : 'desktop-chat')
    await page.evaluate(() => localStorage.setItem('camelid.selectedConversationId', 'checklist'))
    await visit()
    await page.click('button[aria-label="Conversation files"]')
    await page.waitForSelector('.conversation-files .output-preview')
    await capture(mobile ? 'mobile-files' : 'desktop-files')
    if (mobile) {
      await page.click('[aria-label="Close conversation files"]')
      await page.click('[aria-label="Edit conversation context"]')
      await page.waitForSelector('.context-modal')
      await capture('mobile-context')
    } else {
      await visit('projects')
      await page.waitForSelector('[aria-label="New chat in Field notes"]')
      await capture('desktop-projects')
    }
    await page.close()
  }
  assert.deepEqual(errors, [], 'Browser runtime errors')
} finally {
  await browser.close()
  await new Promise(done => server.close(done))
}
