// Explicit real-model regression for an agent promising files without writing.
// Run on the test host with an isolated engine store; never a user's project.
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, realpathSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, dirname, basename } from 'node:path'
import { randomUUID } from 'node:crypto'
import { createServer } from 'node:http'
import { launchBrowser } from './lib/launch-browser.mjs'
const base = process.env.CAMELID_CODING_LIVE_URL
assert.ok(base && ['127.0.0.1','localhost','[::1]'].includes(new URL(base).hostname), 'Set an isolated loopback engine URL.')
const out = resolve(process.env.CAMELID_CODING_LIVE_OUT || '../target/coding-site-live')
mkdirSync(out, { recursive: true })
const prior = process.env.CAMELID_CODING_SITE_RESUME ? JSON.parse(readFileSync(process.env.CAMELID_CODING_SITE_RESUME, 'utf8')) : null
const workspace = prior ? realpathSync(prior.config.workspace) : mkdtempSync(join(tmpdir(), 'camelid-tiny-tasks-'))
assert.equal(realpathSync(dirname(workspace)), realpathSync(tmpdir()), 'Only this harness’s temporary fixtures may be resumed.')
assert.ok(basename(workspace).startsWith('camelid-tiny-tasks-'))
const paths = ['index.html','style.css','app.js']
const route = '/api/agent/coding/sessions'
async function request(path, method = 'GET', body) {
 const response = await fetch(base + path, { method, headers: { 'Content-Type':'application/json', Origin:base }, ...(body ? { body:JSON.stringify(body) } : {}), signal:AbortSignal.timeout(15000) })
 const data = await response.json(); assert.ok(response.ok, `${response.status}: ${JSON.stringify(data)}`); return data
}
const health = await request('/v1/health')
const goal = 'Build a small task-list website called Tiny Tasks in this project folder. Create the actual index.html, style.css, and app.js files using plain HTML, CSS, and JavaScript. Support adding a task by button or Enter, ignore blank tasks, clear the input after adding, toggle complete and incomplete with completed text crossed out, delete an individual task, show an unfinished-task counter and an empty state, and restore tasks from localStorage after refresh. Render task text safely, and make the layout responsive. Check the file links and reread your files. Commands are disabled; say which checks you actually performed and which tests remain. Keep the implementation small.'
if (prior) {
 assert.match(prior.id,/^[a-f0-9]{32}$/)
 const current = await request(`${route}/${prior.id}`)
 assert.equal(realpathSync(current.config.workspace),workspace,'Saved receipt must match the live fixture workspace')
 assert.equal(current.config.model_id,health.active_model_id)
 assert.equal(current.turns[0].user,goal,'Only a session created by this harness may be resumed')
 assert.ok(!['running','paused','waiting_approval','stopping'].includes(current.phase),'Wait for the fixture run to stop before resuming')
}
let session = prior ? await request(`${route}/${prior.id}/messages`, 'POST', { message: process.env.CAMELID_CODING_SITE_FEEDBACK || 'Inspect the generated site, finish its remaining requirements, and accurately summarize your checks.', message_id:randomUUID().replaceAll('-','') }) : await request(route, 'POST', { workspace, goal, message_id:randomUUID().replaceAll('-',''), model_id:health.active_model_id, allow_commands:false, max_steps:32, max_tokens:2048 })
const id = session.id, decisions = [], started = Date.now()
console.log(JSON.stringify({ session:id, workspace, model:health.active_model_id }))
let seq = -1
try {
 while (['running','paused','waiting_approval','stopping'].includes(session.phase)) {
  if (session.seq !== seq) { console.log(JSON.stringify({ seq:session.seq, phase:session.phase, action:session.agents.lead?.action, reviews:session.reviews.length })); seq = session.seq }
  if (session.approval && !decisions.some(d => d.id === session.approval.id)) {
   const a = session.approval, review = a.detail.review
   const approved = Boolean(review && paths.includes(review.path) && typeof review.after === 'string' && review.after.length <= 64000 && decisions.filter(d=>d.approved).length < 9)
   decisions.push({ id:a.id, path:review?.path, approved })
   await request(`${route}/${id}/approvals/${a.id}`, 'POST', { approved })
  }
  assert.ok(Date.now() - started < 15*60*1000, 'Model run exceeded 15 minutes')
  await new Promise(resolve=>setTimeout(resolve,1000))
  session = await request(`${route}/${id}`)
 }
 assert.equal(session.phase,'completed',session.error)
 for (const path of paths) {
  assert.ok(existsSync(join(workspace,path)),`Agent did not create ${path}`)
  assert.ok(session.reviews.some(r=>r.path===path && r.status==='applied'),`No applied review for ${path}`)
 }
 const html = readFileSync(join(workspace,'index.html'),'utf8')
 assert.match(html,/style\.css/);assert.match(html,/app\.js/)
 console.log('All three project files have actual approved writes. Starting independent browser verification.')
 const server = createServer((req,res)=>{
  const name = new URL(req.url,'http://localhost').pathname.slice(1) || 'index.html'
  if (!paths.includes(name)) { res.writeHead(404);res.end();return }
  res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.css')?'text/css':'text/javascript'});res.end(readFileSync(join(workspace,name)))
 })
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
 const site = `http://127.0.0.1:${server.address().port}`
 const browser = await launchBrowser({headless:true})
 try {
  const page = await browser.newPage(), errors=[]
  page.on('pageerror',error=>errors.push(error.message))
  page.on('dialog',async dialog=>{errors.push('Unexpected browser dialog: '+dialog.message());await dialog.dismiss()})
  await page.setRequestInterception(true)
  page.on('request',request=>request.url().startsWith(site+'/') ? request.continue() : request.abort())
  await page.setViewport({width:1200,height:850})
  await page.goto(site,{waitUntil:'networkidle2'})
  assert.deepEqual(errors,[], 'The generated site must load without JavaScript errors')
  const input = 'input:not([type]),input[type="text"]'
  await page.waitForSelector(input)
  await page.type(input,'First task')
  const add = await page.evaluateHandle(()=>[...document.querySelectorAll('button')].find(b=>/^add(?:\s|$)/i.test(b.textContent.trim())))
  assert.ok(add.asElement(),'An Add button is required');await add.asElement().click();await add.dispose()
  await page.waitForFunction(()=>document.body.innerText.includes('First task'))
  assert.equal(await page.$eval(input,e=>e.value),'')
  await page.type(input,'<img src=x onerror=alert(1)>');await page.keyboard.press('Enter')
  await page.waitForFunction(()=>document.body.innerText.includes('<img src=x onerror=alert(1)>'))
  assert.equal(await page.$('img'),null,'Task input must not create HTML')
  const beforeBlank = await page.$eval('body',e=>e.innerText)
  await page.type(input,'   ');await page.keyboard.press('Enter')
  assert.equal(await page.$eval('body',e=>e.innerText),beforeBlank,'Blank tasks must be ignored')
  await page.reload({waitUntil:'networkidle2'})
  await page.waitForFunction(()=>document.body.innerText.includes('First task') && document.body.innerText.includes('<img src=x onerror=alert(1)>'))
  await page.screenshot({path:join(out,'tiny-tasks-desktop.png'),fullPage:true})
  await page.setViewport({width:390,height:850})
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,'Mobile horizontal overflow')
  await page.screenshot({path:join(out,'tiny-tasks-mobile.png'),fullPage:true})
  const clickTaskControl = async (task, action) => {
   const handle = await page.evaluateHandle((task, action)=>{
    const row = [...document.querySelectorAll('li')].find(e=>e.textContent.includes(task))
    return row && [...row.querySelectorAll('button')].find(b=>(action==='toggle'?/complete|toggle|done/i:/delete|remove/i).test(b.textContent))
   },task,action)
   assert.ok(handle.asElement(),`Missing ${action} control for ${task}`);await handle.asElement().click();await handle.dispose()
  }
  const isCrossedOut = () => page.evaluate(()=>{
   const row=[...document.querySelectorAll('li')].find(e=>e.textContent.includes('First task'))
   return row && [row,...row.querySelectorAll('*')].some(e=>e.textContent.includes('First task') && getComputedStyle(e).textDecorationLine.includes('line-through'))
  })
  assert.equal(await isCrossedOut(),false,'A new task must not be crossed out')
  await clickTaskControl('First task','toggle')
  assert.equal(await isCrossedOut(),true,'Completed task must be crossed out')
  assert.match(await page.$eval('body',e=>e.innerText),/(?:unfinished|remaining|pending|left)[^\n\d]*1\b|\b1\s+(?:tasks? )?(?:left|remaining|unfinished)/i,'Unfinished count must become one')
  await clickTaskControl('First task','toggle');assert.equal(await isCrossedOut(),false)
  await clickTaskControl('First task','delete')
  assert.equal(await page.evaluate(()=>document.body.innerText.includes('First task')),false)
  assert.equal(await page.evaluate(()=>document.body.innerText.includes('<img src=x onerror=alert(1)>')),true,'Deleting one row must preserve the other')
  await clickTaskControl('<img src=x onerror=alert(1)>','delete')
  assert.match(await page.$eval('body',e=>e.innerText),/no tasks|no .*tasks|all caught up|nothing|empty|add.*first/i,'An empty-state message is required')
  assert.deepEqual(errors,[])
  console.log('Independent browser checks passed: real HTML/CSS/JS links, add by button and Enter, blank rejection, input clearing, safe task text, reload persistence, completion styling and reversal, unfinished count, individual deletion, empty state, and mobile width.')
 } finally { await browser.close();await new Promise(resolve=>server.close(resolve)) }
} catch (error) {
 if (['running','paused','waiting_approval','stopping'].includes(session.phase)) await request(`${route}/${id}/control`,'POST',{action:'stop'}).catch(()=>{})
 throw error
} finally {
 writeFileSync(join(out,'session.json'),JSON.stringify(session,null,2))
 writeFileSync(join(out,'decisions.json'),JSON.stringify(decisions,null,2))
}
