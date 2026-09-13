// Wiring evidence with deterministic HTTP fixtures, not a model capability claim.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { readFileSync, existsSync, mkdirSync } from 'node:fs'
import { extname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { launchBrowser } from './lib/launch-browser.mjs'
const root = fileURLToPath(new URL('../..', import.meta.url))
const dist = resolve(root, 'frontend/dist')
const out = resolve(root, 'target/coding-browser')
mkdirSync(out, { recursive: true })
const ledger = JSON.parse(readFileSync(resolve(root, 'ledger/camelid-ledger.json'), 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map(r => ({ ...r.contract, tool_capable: true })) }
const model = 'qwen3_0_6b_instruct_q8_0', filename = 'Qwen3-0.6B-Q8_0.gguf'
const sessionId = 'a'.repeat(32), reviewId = 'b'.repeat(32)
const streams = new Set(), requests = [], errors = [], folderRequests = []
let session = null, approvals = 0, creates = 0, commands = 0
const fileReview = { id: reviewId, workspace: '/project', path: 'src/example.js', before: 'export const answer = 1', after: 'export const answer = 2', diff: '- export const answer = 1\n+ export const answer = 2', status: 'pending', source: 'Coding · Lead', before_bytes: 23, after_bytes: 23, created_at: Date.now() }
function json(res, value, status = 200) { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)) }
function publish() { session.seq++; session.updated_at = Date.now(); const event = `id: ${session.seq}\nevent: coding\ndata: ${JSON.stringify(session)}\n\n`; for (const res of streams) res.write(event) }
async function body(req) { const chunks=[]; for await (const chunk of req) chunks.push(chunk); return JSON.parse(Buffer.concat(chunks).toString() || '{}') }
const server = createServer(async (req,res) => {
 try {
  const path = new URL(req.url, 'http://localhost').pathname
  if (path === '/api/agent/workspace/browse') return json(res, { path: new URL(req.url, 'http://localhost').searchParams.get('path') || '/project', parent: '/', has_roots: false, separator: '/', entries: [], truncated: false })
  if (path === '/api/agent/coding/folders' && req.method === 'POST') {
   const data = await body(req); folderRequests.push(data)
   if (data.name === 'Existing') return json(res, { error: { message: 'A file or folder with that name already exists.' } }, 409)
   return json(res, { path: data.parent + '/' + data.name }, 201)
  }
  if (path === '/v1/health') return json(res, { ok: true, engine: 'camelid', api_surface: 'full', backend: 'llama', model_family: 'qwen3', loaded_now: true, generation_ready: true, active_model_id: model, active_context_length: 4096, max_prompt_tokens: 4096, max_generation_tokens: 8192 })
  if (path === '/v1/models') return json(res, { object: 'list', data: [{ id: model, object: 'model', owned_by: 'camelid', meta: { n_ctx_train: 32768, n_params: 600000000, size: 639446688 } }] })
  if (path === '/api/capabilities') return json(res, capabilities)
  if (path === '/api/models/current') return json(res, { id: model, path: 'models/' + filename, gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } }, tokenizer: { status: 'available' } })
  if (path === '/api/models/local') return json(res, { models_dir: 'models', models: [{ filename, size_bytes: 639446688, architecture: 'qwen3', quantization: 'Q8_0', admitted: true, oracle_qualified: true, chat_capable: true, generation_capable: true, context_length: 32768, lane_class: 'supported' }] })
  if (path === '/api/models/catalog/downloads') return json(res, [])
  if (path === '/api/mcp/connections') return json(res, { connections: [] })
  if (path === '/api/agent/coding/sessions' && req.method === 'GET') return json(res, { tool_capable_model: model, sessions: session ? [{ id: session.id, title: session.title, phase: session.phase, updated_at: session.updated_at, workspace: '/project' }] : [] })
  if (path === '/api/agent/coding/sessions' && req.method === 'POST') {
   const data = await body(req); requests.push(data); creates++
   session = { id: sessionId, run_id: 'c'.repeat(32), title: data.goal, config: { workspace: data.workspace, model_id: model, project_id: data.project_id, allow_commands: data.allow_commands }, seq: 1, phase: 'waiting_approval', updated_at: Date.now(), error: '', turns: [{ user: data.goal, assistant: '', outcome: 'running' }], agents: {
    lead: { id:'lead', parent_id:null, goal:data.goal, status:'waiting_approval', action:'write_file(src/example.js)', files:['src/example.js'], output:'' },
    explorer: { id:'explorer', parent_id:'lead', goal:'Find the answer definition', status:'done', action:'Read src/example.js', files:['src/example.js'], output:'The answer is defined in src/example.js.\n```js\nexport const answer = 1\n```' },
   }, plan:[{text:'Inspect the definition',status:'done'},{text:'Update the answer',status:'in_progress'}], reviews:[{...fileReview}], events:[{seq:1,time:Date.now(),run_id:'c'.repeat(32),agent_id:'explorer',kind:'tool.result',detail:{tool:'read_file',ok:true,content:'export const answer = 1'}}], approval:{id:'approval-file',tool:'write_file',detail:{review:{...fileReview}}} }
   return json(res, session)
  }
  if (path === `/api/agent/coding/sessions/${sessionId}/events`) {
   res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache'}); res.write(`event: coding\ndata: ${JSON.stringify(session)}\n\n`); streams.add(res); res.on('close',()=>streams.delete(res)); return
  }
  if (path === `/api/agent/coding/sessions/${sessionId}` && req.method === 'GET') return json(res, session)
  if (path === `/api/agent/coding/sessions/${sessionId}/control`) {
   const data=await body(req); requests.push(data)
   session.phase=data.action==='pause'?'paused':data.action==='stop'?'cancelled':'running'
   if (data.action==='stop') { session.agents.lead.status='cancelled'; session.approval=null }
   publish(); return json(res,session)
  }
  if (path.startsWith(`/api/agent/coding/sessions/${sessionId}/approvals/`)) {
   const decision=await body(req); requests.push(decision); approvals++
   assert.ok(session.approval, 'only pending approvals can be decided')
   if (session.approval.tool==='run_shell') {
    if (decision.approved) commands++
    session.phase='completed'; session.agents.lead.status='done'; session.turns.at(-1).assistant=decision.approved?'The command completed successfully.\n```sh\necho coding-ok\n```':'I did not execute the denied command.'
   } else {
    fileReview.status=decision.approved?'applied':'rejected'; session.reviews=[{...fileReview}]; session.phase='running'; session.agents.lead.status='working'; session.agents.lead.action='Preparing verification'
   }
   session.approval=null; publish(); return json(res,{accepted:true})
  }
  if (path === `/api/agent/coding/sessions/${sessionId}/messages`) {
   const data=await body(req);requests.push(data);session.run_id='d'.repeat(32);session.phase='waiting_approval';session.turns.push({user:data.message,assistant:'',outcome:'running'});session.agents.lead.status='waiting_approval';session.approval={id:'approval-command',tool:'run_shell',detail:{command:'echo coding-ok',workspace:'/project',execution:'Runs with your account permissions.',timeout_seconds:30}};publish();return json(res,session)
  }
  if (path===`/api/changes/${reviewId}/undo`) {fileReview.status='undone';return json(res,fileReview)}
  if (path===`/api/changes/${reviewId}`) return json(res,fileReview)
  if (path==='/api/changes') return json(res,{reviews:[fileReview]})
  if (path==='/api/telemetry/stream') {res.writeHead(200,{'Content-Type':'text/event-stream'});res.end();return}
  if (path.startsWith('/api/') || path.startsWith('/v1/')) return json(res, {})
  const file=resolve(dist,'.'+(path==='/'?'/index.html':path))
  if (!file.startsWith(dist+'/') || !existsSync(file)) {res.writeHead(404);res.end();return}
  const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2','.woff':'font/woff'}
  res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream'});res.end(readFileSync(file))
 } catch (e) {errors.push(e.message);json(res,{error:{message:e.message}},500)}
})
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
const browser=await launchBrowser({headless:true})
const page=await browser.newPage();page.on('pageerror',e=>{ errors.push(e.message); console.error(e.stack) })
const clickText=async(text,selector='button')=>{
 const handle=await page.evaluateHandle((selector,text)=>[...document.querySelectorAll(selector)].find(e=>e.textContent.trim()===text),selector,text)
 const element=handle.asElement();assert.ok(element,`Missing ${text}`);await element.click();await handle.dispose()
}
try {
 await page.setViewport({width:1440,height:1050,deviceScaleFactor:1})
 await page.evaluateOnNewDocument(()=>{localStorage.setItem('camelid.sidebarCollapsed','true');localStorage.setItem('camelid.theme','dark')})
 await page.goto(`http://127.0.0.1:${server.address().port}`,{waitUntil:'networkidle2'})
 await clickText('Code','.coding-mode-switch button')
 await page.waitForSelector('#coding-folder');await page.type('#coding-folder','/project')
 await clickText('Browse');await page.waitForFunction(()=>!document.querySelector('.folder-picker__up')?.disabled)
 await clickText('New folder');await page.waitForSelector('#project-folder-name')
 assert.equal(await page.evaluate(()=>[...document.querySelectorAll('button')].find(b=>b.textContent==='Create & use').disabled),true)
 await page.type('#project-folder-name','Existing');await page.keyboard.press('Enter')
 await page.waitForFunction(()=>document.querySelector('.folder-picker__create [role="alert"]')?.textContent.includes('already exists'))
 await page.$eval('#project-folder-name',input=>input.select());await page.keyboard.press('Backspace');await page.type('#project-folder-name','Tiny Tasks')
 await page.setViewport({width:390,height:1050,deviceScaleFactor:1})
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,'folder creation mobile overflow')
 await page.screenshot({path:resolve(out,'coding-new-folder.png'),fullPage:true})
 await clickText('Create & use')
 await page.waitForFunction(()=>!document.querySelector('.folder-picker') && document.querySelector('#coding-folder')?.value==='/project/Tiny Tasks')
 assert.deepEqual(folderRequests,[{parent:'/project',name:'Existing'},{parent:'/project',name:'Tiny Tasks'}])
 await page.setViewport({width:1440,height:1050,deviceScaleFactor:1})
 await page.click('.coding-command-choice input')
 await page.type('[aria-label="Message coding agents"]','Update the answer and verify it.')
 await page.waitForFunction(()=>!document.querySelector('[aria-label="Start coding"]').disabled)
 await page.click('[aria-label="Start coding"]')
 await page.waitForSelector('.coding-approval')
 assert.equal(await page.$('.coding-conversation [data-streaming-state="active"]'),null,'approval is not a streaming model response')
 assert.equal(creates,1);assert.equal(requests[0].allow_commands,true)
 assert.equal(requests[0].workspace,'/project/Tiny Tasks')
 assert.equal(requests[0].model_id,model);assert.match(requests[0].message_id,/^[a-f0-9]{32}$/)
 await clickText('Explorer', '.coding-agent strong').catch(async()=>{await page.click('.coding-agent:nth-child(2)')})
 await page.waitForFunction(()=>document.querySelector('.coding-agent-detail')?.textContent.includes('Read-only helper'))
 await page.screenshot({path:resolve(out,'coding-approval-dark.png'),fullPage:true})
 assert.equal(await page.$('.coding-conversation pre'), null, 'source stays out of the conversation')
 assert.equal(await page.$('.coding-diff'), null, 'diff is collapsed by default')
 await clickText('Review file change')
 await page.waitForSelector('.coding-team .coding-diff')
 assert.ok(await page.$('.coding-conversation .cxturn--user'), 'opening a review preserves Chat turns')
 await clickText('Approve & apply')
 await page.waitForFunction(()=>!document.querySelector('.coding-approval'))
 assert.equal(approvals,1)
 await clickText('Pause');await page.waitForFunction(()=>document.querySelector('.coding-status.is-paused'))
 await clickText('Chat','.coding-mode-switch button');await page.waitForSelector('.coding-background')
 await clickText('Open coding session');await page.waitForSelector('.coding-workspace')
 await clickText('Resume');await clickText('Stop')
 await page.waitForFunction(()=>document.querySelector('.coding-status.is-cancelled'))
 await page.reload({waitUntil:'networkidle2'})
 await page.waitForFunction(()=>document.querySelector('.coding-status.is-cancelled'))
 assert.equal(creates,1,'reload must not create/replay a run');assert.equal(approvals,1)
 await page.type('[aria-label="Message coding agents"]','Run the approved verification.')
 await page.click('[aria-label="Send coding follow-up"]')
 await page.waitForSelector('.coding-approval-summary')
 assert.equal(await page.evaluate(()=>document.querySelector('.coding-conversation').textContent.includes('echo coding-ok')),false)
 await clickText('Review command')
 await page.waitForFunction(()=>document.querySelector('.coding-team .coding-approval')?.textContent.includes('echo coding-ok'))
 assert.equal(commands,0,'command must not execute before approval')
 await clickText('Allow command once');await page.waitForFunction(()=>document.querySelector('.coding-status.is-completed'))
 assert.equal(commands,1)
 await page.waitForFunction(()=>document.querySelector('.coding-conversation')?.textContent.includes('The command completed successfully.'))
 assert.equal(await page.$('.coding-conversation pre'), null)
 assert.equal(await page.evaluate(()=>document.querySelector('.coding-conversation').textContent.includes('echo coding-ok')),false)
 await clickText('View code in sidebar')
 await page.waitForSelector('.coding-team .coding-snippets details:not([open])')
 await page.click('.coding-snippets summary')
 assert.equal(await page.$eval('.coding-snippets details',e=>e.open),true)
 assert.match(await page.$eval('.coding-snippets pre',e=>e.textContent),/echo coding-ok/)
 await clickText('← Back to activity')
 for (const theme of ['dark','light']) {
  await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme)
  for (const width of [1440,1024,768,390,320]) {
   await page.setViewport({width,height:1050,deviceScaleFactor:1})
   await page.waitForFunction(() => [...document.getAnimations()].every(animation => animation.effect?.getComputedTiming().iterations === Infinity || animation.playState !== 'running'))
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`${theme} ${width}: horizontal overflow`)
   if ([1440,390].includes(width)) await page.screenshot({path:resolve(out,`coding-${theme}-${width}.png`),fullPage:true})
  }
 }
 await page.setViewport({width:1440,height:1050,deviceScaleFactor:1})
 await page.click('.coding-proposals > button')
 await page.click('.coding-review-list button')
 await page.waitForSelector('.coding-team .coding-review .coding-diff')
 await clickText('Undo change');await page.waitForFunction(()=>document.querySelector('.coding-review')?.textContent.includes('undone'))
 assert.equal(fileReview.status,'undone')
 await page.reload({waitUntil:'networkidle2'})
 await page.waitForFunction(()=>document.querySelector('.coding-proposals')?.textContent.includes('0 applied'))
 assert.deepEqual(errors,[])
 console.log('Coding browser smoke passed: shared Chat turns, description-only conversation, collapsed sidebar diffs/snippets/commands, real UI controls, versioned events, agent assignments, file/command approvals, pause/resume/stop, background navigation, reload without replay, follow-up, undo, and dark/light responsive layouts.')
} catch(e) {
 await page.screenshot({path:resolve(out,'failure.png'),fullPage:true}).catch(()=>{})
 console.error(errors);throw e
} finally {
 await browser.close();for(const res of streams)res.end();await new Promise(resolve=>server.close(resolve))
}
