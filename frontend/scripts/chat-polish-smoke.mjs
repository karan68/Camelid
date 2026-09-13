import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'
import { conversationFiles } from '../src/lib/conversationFiles.js'
import { toolActivityGroups } from '../src/lib/toolActivity.js'
import { codeFences } from '../src/lib/codeFences.js'
import { normalizeStoredMessage } from '../src/lib/conversationStorage.js'

const call = { id: 'reused', function: { name: 'mcp_echo', arguments: '{"text":"hello"}' } }
const messages = [
  { id: 'a', role: 'assistant', mcp_managed: true, tool_calls: [call], content: '```json\r\n{"a":1}\r\n```' },
  { id: 'r1', role: 'tool', tool_call_id: 'reused', content: 'first', mcp: { status: 'complete', duration_ms: 400 } },
  { id: 'b', role: 'assistant', mcp_managed: true, tool_calls: [call], content: '' },
  { id: 'r2', role: 'tool', tool_call_id: 'reused', content: 'second', mcp: { status: 'denied' } },
  { id: 'orphan', role: 'tool', tool_call_id: 'unknown', content: 'orphan' },
]
const groups = toolActivityGroups(messages)
assert.equal(groups.groups.get('a')[0].result.content, 'first')
assert.equal(groups.groups.get('b')[0].result.content, 'second')
assert.equal(groups.pairedResults.has('orphan'), false)
assert.equal(toolActivityGroups([{ ...messages[0], tool_calls: [call, call] }, messages[1]]).pairedResults.size, 0, 'duplicate call IDs cannot steal a result')
assert.equal(normalizeStoredMessage(messages[1]).mcp.duration_ms, 400)
assert.equal(conversationFiles([{id:'json',role:'assistant',content:'{"ok":true}',output_format:'json'}])[0].preview, 'json', 'structured replies remain files after reload')
const files = conversationFiles([...messages, ...Array.from({length:70}, (_,i) => ({id:'u'+i, role:'user', content:'```html\nNo file\n```'}))])
assert.equal(files.length, 1, 'older assistant files stay available; user quotes are not generated files')
assert.equal(files[0].text, '{"a":1}\n')
assert.equal(files[0].id, 'a:0')
assert.equal(conversationFiles([{id:'s',role:'assistant',streaming:true,content:'```js\nconst a=1\n```\n```js\nconst b='}]).length, 1, 'open streaming fences are unavailable')
assert.equal(codeFences('```python print("inline")\nprint("next")\n```')[0].code, 'print("inline")\nprint("next")\n')
assert.equal(conversationFiles([{id:'t',role:'tool',content:JSON.stringify({content:[{type:'resource',resource:{uri:'mcp://files/report.csv',mimeType:'text/csv',text:'a,b\n1,2'}}]})}])[0].name,'report.csv')

const vite = await createServer({ server: { middlewareMode: true }, appType: 'custom' })
try {
  const { ContextMeter } = await vite.ssrLoadModule('/src/components/chat/ContextMeter.jsx')
  const { ToolActivityCard } = await vite.ssrLoadModule('/src/components/mcp/ToolActivityCard.jsx')
  for (const [percent, pressure] of [[79,'low'],[80,'warning'],[94,'warning'],[95,'high'],[100,'high']]) {
    const html=renderToStaticMarkup(React.createElement(ContextMeter,{contextLength:1000,promptTokens:percent*10,reservedTokens:0}))
    assert.ok(html.includes('pressure-'+pressure), 'threshold '+percent)
    assert.match(html,/ctxmeter__fill--reserved/, 'reply reservation keeps its own segment')
  }
  const completed=renderToStaticMarkup(React.createElement(ToolActivityCard,{call,result:messages[1]}))
  assert.match(completed,/Completed/);assert.match(completed,/400 ms/);assert.doesNotMatch(completed,/Allow once/)
  const props={call,live:{phase:'approval',receiptId:'expected',tool:'echo'},approval:{id:'different'}}
  assert.doesNotMatch(renderToStaticMarkup(React.createElement(ToolActivityCard,props)),/Allow once/,'a different approval cannot be acted on from this card')
  assert.match(renderToStaticMarkup(React.createElement(ToolActivityCard,{...props,approval:{id:'expected'}})),/Allow once/)
} finally { await vite.close() }
console.log('Chat polish smoke passed: context thresholds, transcript files, streaming, persisted timings, call identity, and approval ownership.')
