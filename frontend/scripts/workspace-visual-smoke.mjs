import { existsSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const executablePath = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].find(existsSync)
if (!executablePath) throw new Error('Chrome or Edge is required for Workspace visual smoke')

const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const browser = await puppeteer.launch({ executablePath, headless: 'new' })

const health = { ok: true, engine: 'camelid', loaded_now: true, generation_ready: true, active_model_id: 'Llama 3.2 3B Instruct', q8_runtime: {}, execution_plan: null, backend: 'llama', model_family: 'llama-family', gemma4_available: false, engine_queue_depth: 0 }
const models = { object: 'list', data: [{ id: 'Llama 3.2 3B Instruct', object: 'model', created: 0, owned_by: 'camelid', meta: { n_ctx_train: 8192, n_params: 3210000000, size: 3421898816 } }] }
const currentModel = { id: 'Llama 3.2 3B Instruct', path: 'C:/models/Llama-3.2-3B-Instruct-Q8_0.gguf', gguf: { metadata: { general: { file_type: 7 } } }, llama_config: {}, llama_tensors: {}, tokenizer: { status: 'available' } }
const local = { models_dir: 'C:/models', models: [{ filename: 'Llama-3.2-3B-Instruct-Q8_0.gguf', size_bytes: 3421898816, architecture: 'llama', quantization: 'Q8_0', tokenizer_kind: 'gpt2_bpe', admitted: true, oracle_qualified: true, chat_capable: true, context_length: 8192, lane_class: 'supported' }] }
const capabilities = {
  engine: 'camelid',
  model_compatibility: [{
    id: 'llama32_3b_instruct_q8_0',
    family: 'llama_bpe_decoder',
    quantization: 'Q8_0',
    status: 'supported_exact_row_smoke',
    tool_capable: true,
  }],
  planned_model_families: [],
  api_features: [],
  support_contract: { current_gate: 'Exact-row fixture for Workspace visual validation.' },
}

async function jsonResponse(request, body, status = 200) {
  await request.respond({
    status,
    contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': '*' },
    body: JSON.stringify(body),
  })
}

try {
  const page = await browser.newPage()
  await page.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    class MockEventSource {
      constructor() { this.listeners = new Map(); this.closed = false }
      addEventListener(type, callback) {
        this.listeners.set(type, callback)
        if (type !== 'workspace') return
        const emit = (delay, payload) => setTimeout(() => this.emit(payload), delay)
        emit(50, { sequence: 1, event: 'session.started', model_id: 'Llama 3.2 3B Instruct' })
        emit(100, { sequence: 2, event: 'tool.call', detail: 'read_file(src/api/mod.rs)' })
        emit(150, { sequence: 3, event: 'tool.result', tool: 'read_file', outcome: 'ok', content: 'Read 64 KB from src/api/mod.rs' })
        emit(200, { sequence: 4, event: 'tool.call', detail: 'edit_file(src/api/mod.rs)' })
        emit(250, { sequence: 5, event: 'approval.required', approval_id: 'approval-visual', tool: 'edit_file', risk: 'write', detail: 'edit_file -> src/api/mod.rs\n  - old validation branch\n  + bounded workspace validation' })
      }
      emit(payload) { if (!this.closed) this.listeners.get('workspace')?.({ data: JSON.stringify(payload) }) }
      close() { this.closed = true }
    }
    globalThis.EventSource = MockEventSource
  })
  await page.setRequestInterception(true)
  const intercepted = []
  page.on('request', async (request) => {
    const url = request.url()
    if (url.includes('127.0.0.1:8181')) intercepted.push(`${request.method()} ${url}`)
    if (url.includes('127.0.0.1:8181') && request.method() === 'OPTIONS') {
      return request.respond({
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
        body: '',
      })
    }
    if (url.endsWith('/v1/health')) return jsonResponse(request, health)
    if (url.endsWith('/v1/models')) return jsonResponse(request, models)
    if (url.endsWith('/api/capabilities')) return jsonResponse(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return jsonResponse(request, [])
    if (url.endsWith('/api/models/current')) return jsonResponse(request, currentModel)
    if (url.endsWith('/api/models/local')) return jsonResponse(request, local)
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      return jsonResponse(request, { id: 'workspace-visual', workspace: 'C:/projects/camelid', model_id: 'Llama 3.2 3B Instruct', state: 'waiting_for_events', max_steps: 12, max_tokens: 800 }, 201)
    }
    if (url.includes('/api/agent/workspace/sessions/workspace-visual/')) return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*' }, body: '' })
    return request.continue()
  })

  for (const viewport of [{ width: 1280, height: 800, name: 'desktop' }, { width: 390, height: 844, name: 'mobile' }]) {
    await page.setViewport(viewport)
    await page.goto('about:blank')
    await page.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
    await page.waitForSelector('.workspace-view')
    await page.type('.workspace-field input', 'C:/projects/camelid')
    await page.type('.workspace-field--goal textarea', 'Inspect the API and make the smallest safe validation repair.')
    const buttons = await page.$$('button')
    const startButton = await Promise.all(buttons.map(async (button) => ({ button, text: await button.evaluate((node) => node.innerText) }))).then((items) => items.find((item) => item.text.includes('Start Workspace'))?.button)
    if (!startButton) throw new Error(`${viewport.name}: Start Workspace button not found`)
    if (await startButton.evaluate((node) => node.disabled)) {
      const diagnostic = await page.evaluate(() => ({
        model: document.querySelector('.workspace-model-line')?.innerText,
        blocked: document.querySelector('.workspace-blocked')?.innerText,
      }))
      throw new Error(`${viewport.name}: earned model did not unlock Workspace ${JSON.stringify({ diagnostic, intercepted })}`)
    }
    await startButton.click()
    await page.waitForSelector('.workspace-approval', { timeout: 5000 })

    const metrics = await page.evaluate(() => {
      const root = document.documentElement
      const modal = document.querySelector('.cx-modal')
      const rail = document.querySelector('.rail')
      const approvalButtons = [...document.querySelectorAll('.workspace-approval__actions .cx-btn')]
      return {
        documentWidth: [root.clientWidth, root.scrollWidth],
        modal: modal ? { left: modal.getBoundingClientRect().left, right: modal.getBoundingClientRect().right, width: modal.getBoundingClientRect().width, scrollWidth: modal.scrollWidth, clientWidth: modal.clientWidth } : null,
        railLeft: rail?.getBoundingClientRect().left,
        buttons: approvalButtons.map((button) => ({ text: button.innerText, height: button.getBoundingClientRect().height, width: button.getBoundingClientRect().width })),
      }
    })
    if (metrics.documentWidth[0] !== metrics.documentWidth[1]) throw new Error(`${viewport.name}: page overflow ${metrics.documentWidth}`)
    if (!metrics.modal || metrics.modal.left < 0 || metrics.modal.right > viewport.width || metrics.modal.scrollWidth > metrics.modal.clientWidth) throw new Error(`${viewport.name}: modal overflow ${JSON.stringify(metrics.modal)}`)
    if (metrics.buttons.some((button) => button.height < 36)) throw new Error(`${viewport.name}: approval control below 36px ${JSON.stringify(metrics.buttons)}`)
    if (viewport.name === 'mobile' && !(metrics.railLeft < 0)) throw new Error(`mobile: navigation rail was not hidden (${metrics.railLeft})`)
    await page.screenshot({ path: `../target/workspace-approval-${viewport.name}.png` })
    console.log(`${viewport.name}: PASS ${JSON.stringify(metrics)}`)
  }
  await page.close()
} finally {
  await browser.close()
}

console.log('workspace-visual-smoke: PASS')
