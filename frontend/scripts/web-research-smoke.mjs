#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import {
  applyWebResearchContext,
  boundWebResearchResult,
  canEnableNativeModelTools,
  classifyWebResearchNeed,
  deriveWebResearchPromptBudget,
  effectiveGenerationTokenLimit,
  extractPromptUrls,
  fitWebResearchContext,
  persistWebResearchEnabled,
  readWebResearchEnabled,
  requestWebResearch,
  webResearchMetadata,
} from '../src/lib/webResearch.js'

const scriptDir = dirname(fileURLToPath(import.meta.url))

const acceptancePrompt = `I want to update an app that I once developed. Build a multi-step Xcode task list.
1. connect to a bluetooth based scale for food measurement ([https://github.com/bburky/smartchef-web-bluetooth/](https://github.com/bburky/smartchef-web-bluetooth/) and [https://github.com/PanamaHitek/SmartScale)2](https://github.com/PanamaHitek/SmartScale\\)2). allow meal assembly mode.`

assert.deepEqual(extractPromptUrls(acceptancePrompt), [
  'https://github.com/bburky/smartchef-web-bluetooth/',
  'https://github.com/PanamaHitek/SmartScale',
], 'the acceptance prompt should recover and deduplicate both canonical GitHub repositories')

const linkedPlan = classifyWebResearchNeed(acceptancePrompt)
assert.equal(linkedPlan.needed, true)
assert.equal(linkedPlan.reason, 'linked_urls')
assert.equal(linkedPlan.urls.length, 2)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: true, modelArchitecture: 'gemma4', certifiedGemma4Tools: true }), false)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: false, modelArchitecture: 'qwen3', certifiedGemma4Tools: true }), false)
assert.equal(canEnableNativeModelTools({ webResearchEnabled: false, modelArchitecture: 'gemma4', certifiedGemma4Tools: true }), true)

assert.deepEqual(extractPromptUrls('Read https://github.com/acme/widgets/tree/feature/web-research'), [
  'https://github.com/acme/widgets/tree/feature/web-research',
], 'GitHub branch refs and paths must survive browser URL normalization')
const mixedPlan = classifyWebResearchNeed('Read https://example.com/spec and also search the web for current alternatives.')
assert.equal(mixedPlan.reason, 'linked_urls_and_search')
assert.deepEqual(mixedPlan.urls, ['https://example.com/spec'])
assert.match(mixedPlan.query, /search the web/)
const linkedReadOnlyPlan = classifyWebResearchNeed('Read the linked docs https://example.com/spec')
assert.equal(linkedReadOnlyPlan.reason, 'linked_urls')
assert.equal(linkedReadOnlyPlan.query, null, 'reading an explicit link must not create a second search query')

for (const ordinary of [
  'Rewrite this paragraph more concisely.',
  'Turn this checklist into ordered steps.',
  'Today I went to the store.',
  'Use the current fitness setup from my draft.',
]) {
  assert.equal(classifyWebResearchNeed(ordinary).needed, false, `ordinary local prompt should not browse: ${ordinary}`)
}

const currentPlan = classifyWebResearchNeed('Search the web for the current Xcode release and cite sources.')
assert.equal(currentPlan.needed, true)
assert.equal(currentPlan.reason, 'explicit_search')
for (const publicWebCue of [
  'Search online for Xcode release notes.',
  'Use the internet to compare these libraries.',
  'Browse online for a current answer.',
  'What is the latest Xcode release?',
  'Show the most recent Xcode information.',
]) {
  assert.equal(classifyWebResearchNeed(publicWebCue).needed, true, `public-web egress must be disclosed before send: ${publicWebCue}`)
}

const originalMessages = [{ role: 'user', content: acceptancePrompt }]
const enriched = applyWebResearchContext(originalMessages, {
  triggered: true,
  reason: 'linked_urls',
  sources: [
    { title: 'SmartChef Web Bluetooth', url: 'https://github.com/bburky/smartchef-web-bluetooth', excerpt: 'UNIQUE_GATT_MARKER' },
    { title: 'SmartScale', url: 'https://github.com/PanamaHitek/SmartScale', excerpt: 'UNIQUE_ADVERTISEMENT_MARKER' },
  ],
})
assert.equal(enriched[0].role, 'system', 'research evidence should be a leading system message')
assert.match(enriched[0].content, /UNTRUSTED EXTERNAL DATA/)
assert.match(enriched[0].content, /UNIQUE_GATT_MARKER/)
assert.match(enriched[0].content, /UNIQUE_ADVERTISEMENT_MARKER/)
assert.match(enriched[0].content, /https:\/\/github\.com\/bburky\/smartchef-web-bluetooth/)
assert.deepEqual(enriched.at(-1), originalMessages[0], 'the original user prompt must remain unchanged')

const bounded = applyWebResearchContext(originalMessages, {
  sources: [
    { title: 'Large A', url: 'https://example.com/a', excerpt: `${'A'.repeat(7_000)}TAIL_A` },
    { title: 'Large B', url: 'https://example.com/b', excerpt: `${'B'.repeat(7_000)}TAIL_B` },
  ],
})
assert.doesNotMatch(bounded[0].content, /TAIL_A|TAIL_B/, 'source excerpts must stay within the Gemma Web UI prompt envelope')
assert.ok(bounded[0].content.length < 12_000, 'combined web evidence must have a hard total character bound')

const balanced = boundWebResearchResult({
  sources: [
    { title: 'Repo one', url: 'https://example.com/one', excerpt: 'A'.repeat(7_000) },
    { title: 'Repo two', url: 'https://example.com/two', excerpt: `${'B'.repeat(3_900)}SECOND_REPO_FORMULA` },
  ],
})
assert.equal(balanced.sources[0].excerpt.length, 4_000)
assert.match(balanced.sources[1].excerpt, /SECOND_REPO_FORMULA/, 'a long first repository must not starve the second source of evidence')

const chunked = boundWebResearchResult({
  sources: [
    {
      title: 'Chunked repository',
      url: 'https://github.com/acme/widgets/tree/feature/web-research',
      chunks: [
        { path: 'docs/overview.md', text: 'generic prose '.repeat(300) },
        { path: 'src/protocol.rs', text: `${'setup '.repeat(300)}QUERY_PROTOCOL_MARKER decoder` },
      ],
    },
  ],
}, { maxExcerptChars: 900, queryText: 'protocol decoder' })
assert.equal(chunked.sources[0].url, 'https://github.com/acme/widgets/tree/feature/web-research')
assert.equal(chunked.sources[0].chunks[0].path, 'src/protocol.rs')
assert.match(chunked.sources[0].chunks[0].text, /QUERY_PROTOCOL_MARKER/, 'query-aware chunks must retain a late relevant section')

const minimumPerSource = boundWebResearchResult({
  sources: [1, 2, 3].map((id) => ({ title: `Source ${id}`, url: `https://example.com/${id}`, excerpt: String(id).repeat(2_000) })),
}, { maxExcerptChars: 768 })
assert.equal(minimumPerSource.sources.length, 3)
assert.ok(minimumPerSource.sources.every((source) => source.chunks[0].text.length >= 256), 'every retained source should receive the named minimum when room permits')
const insufficientMinimum = boundWebResearchResult({
  sources: [1, 2, 3].map((id) => ({ title: `Source ${id}`, url: `https://example.com/${id}`, excerpt: String(id).repeat(2_000) })),
}, { maxExcerptChars: 500 })
assert.equal(insufficientMinimum.sources.length, 1, 'insufficient room should omit whole trailing sources instead of listing empty evidence')
assert.match(insufficientMinimum.warnings.at(-1), /1 of 3/)
const belowMinimum = boundWebResearchResult({
  sources: [{ title: 'Too tight', url: 'https://example.com/tight', excerpt: 'x'.repeat(2_000) }],
}, { maxExcerptChars: 255 })
assert.equal(belowMinimum.sources.length, 0, 'a source must not be listed when it cannot receive the named minimum evidence space')

const relevantAmongDistractors = boundWebResearchResult({
  sources: [{
    title: 'Many chunks',
    url: 'https://example.com/many',
    chunks: [
      ...Array.from({ length: 8 }, (_, index) => ({ path: `docs/${index}.md`, text: 'distractor '.repeat(100) })),
      { path: 'src/decoder.rs', text: `${'setup '.repeat(100)}HIGH_VALUE_DECODER` },
    ],
  }],
}, { maxExcerptChars: 256, queryText: 'decoder' })
assert.match(relevantAmongDistractors.sources[0].chunks[0].text, /HIGH_VALUE_DECODER/, 'the best query chunk must retain meaningful space even with many distractors')

const fitted = fitWebResearchContext(originalMessages, {
  triggered: true,
  sources: [
    {
      title: `A title with JSON escapes ${'\\"'.repeat(300)}`,
      url: 'https://example.com/a-long-source-url',
      excerpt: `${'\\"'.repeat(5_000)}CRITICAL_TAIL`,
    },
  ],
}, {
  maxPromptTokens: 700,
  estimateTokenCount: (messages) => messages.reduce((total, message) => total + String(message.content || '').length, 0),
})
assert.ok(fitted.messages.reduce((total, message) => total + String(message.content || '').length, 0) <= 700, 'the complete injected system message must fit, including JSON escaping and metadata')
assert.ok(fitted.research.warnings.length > 0 || fitted.research.sources.length > 0)
const unboundedFit = fitWebResearchContext(originalMessages, {
  triggered: true,
  sources: [{ title: 'Usable', url: 'https://example.com/usable', excerpt: 'EVIDENCE_SURVIVES_WITHOUT_MODEL_CONTEXT_METADATA' }],
}, { maxPromptTokens: null, estimateTokenCount: () => Number.MAX_SAFE_INTEGER })
assert.match(unboundedFit.messages[0].content, /EVIDENCE_SURVIVES_WITHOUT_MODEL_CONTEXT_METADATA/, 'missing context metadata must not be treated as a zero-token prompt budget')

const estimatedTokens = (messages) => messages.reduce((total, message) => total + Math.ceil(String(message.content || '').length / 4) + 4, 3)
const budgetHistory = [
  { role: 'system', content: 'Configured system policy '.repeat(20) },
  { role: 'user', content: 'Earlier history '.repeat(80) },
  { role: 'assistant', content: 'Earlier response '.repeat(80) },
  { role: 'user', content: acceptancePrompt },
]
const runtimeBudget = deriveWebResearchPromptBudget({
  contextLength: 4_096,
  serverMaxPromptTokens: 4_096,
  serverMaxGenerationTokens: 8_192,
  requestedMaxTokens: 1_536,
  messages: budgetHistory,
  estimateTokenCount: estimatedTokens,
})
assert.equal(runtimeBudget.replyReserve, 1_536, 'research must preserve the full requested reply allowance when it fits')
assert.equal(effectiveGenerationTokenLimit(9_000, 4_096), 4_096, 'the same server generation ceiling used for budgeting must cap the outgoing request')
const runtimeFit = fitWebResearchContext(budgetHistory, {
  triggered: true,
  sources: [{ title: `Escaped ${'\\"'.repeat(80)}`, url: 'https://example.com/runtime', excerpt: `${'evidence\\"'.repeat(1_500)}TAIL` }],
}, {
  maxPromptTokens: runtimeBudget.maxPromptTokens,
  estimateTokenCount: estimatedTokens,
  queryText: 'runtime evidence',
})
assert.ok(
  estimatedTokens(runtimeFit.messages) + runtimeBudget.replyReserve + runtimeBudget.safetyMargin <= runtimeBudget.contextLength,
  'history, policy, evidence JSON/template overhead, safety, and reply reserve must fit the active runtime context',
)
const largerRuntimeBudget = deriveWebResearchPromptBudget({
  contextLength: 32_768,
  requestedMaxTokens: 3_072,
  messages: budgetHistory,
  estimateTokenCount: estimatedTokens,
})
assert.ok(largerRuntimeBudget.maxPromptTokens > runtimeBudget.maxPromptTokens, 'evidence room must derive from the active runtime context instead of a fixed small-model envelope')

const noSources = applyWebResearchContext(originalMessages, { triggered: true, sources: [] })
assert.equal(noSources, originalMessages, 'failed research must not inject stale or fabricated evidence')

const metadata = webResearchMetadata({
  triggered: true,
  reason: 'linked_urls',
  sources: [
    { title: 'Safe', url: 'https://example.com/source', excerpt: 'usable evidence' },
    { title: 'Unsafe', url: 'javascript:alert(1)', excerpt: 'must not display' },
  ],
  warnings: ['one linked page was unavailable'],
})
assert.deepEqual(metadata.sources, [{ title: 'Safe', url: 'https://example.com/source' }])
assert.deepEqual(metadata.warnings, ['one linked page was unavailable'])
assert.doesNotMatch(JSON.stringify(metadata), /usable evidence|excerpt|chunks|path/, 'persisted source metadata must never contain fetched text or chunk paths')
const fittedMetadata = webResearchMetadata(insufficientMinimum)
assert.equal(fittedMetadata.sources.length, insufficientMinimum.sources.length, 'persisted provenance must list only sources retained by prompt fitting')
const emptyEvidenceMetadata = webResearchMetadata({
  triggered: true,
  sources: [{ title: 'Not actually read', url: 'https://example.com/empty', excerpt: '' }],
  warnings: [],
})
assert.deepEqual(emptyEvidenceMetadata.sources, [], 'a URL with no injected excerpt must never be displayed as read provenance')
assert.match(emptyEvidenceMetadata.warnings[0], /no usable text excerpts/)

assert.equal(readWebResearchEnabled(), true, 'Web Auto should default on so the pasted acceptance prompt works without setup')

const stored = new Map()
globalThis.window = {
  localStorage: {
    getItem: (key) => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
    removeItem: (key) => stored.delete(key),
  },
}
persistWebResearchEnabled(false)
assert.equal(readWebResearchEnabled(), false, 'turning Web off must persist the zero-research choice')
persistWebResearchEnabled(true)
assert.equal(readWebResearchEnabled(), true, 'turning Web back on must persist Auto mode')
delete globalThis.window

const nativeFetch = globalThis.fetch
try {
  globalThis.fetch = async () => new Response('<html>proxy error</html>', { status: 200, headers: { 'content-type': 'text/html' } })
  await assert.rejects(
    () => requestWebResearch('http://127.0.0.1:8181', 'search the web'),
    /invalid response/,
    'a malformed 2xx response must not silently masquerade as a skipped lookup',
  )

  let requested = null
  globalThis.fetch = async (url, init) => {
    requested = { url, init }
    return new Response(JSON.stringify({ status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })
  }
  const skipped = await requestWebResearch('http://127.0.0.1:8181/', 'Rewrite this sentence.')
  assert.equal(skipped.triggered, false)
  assert.equal(requested.url, 'http://127.0.0.1:8181/api/web/research')
  assert.deepEqual(JSON.parse(requested.init.body), { prompt: 'Rewrite this sentence.' })

  globalThis.fetch = async (_url, init) => new Promise((resolve, reject) => {
    init.signal.addEventListener('abort', () => {
      const error = new Error('aborted')
      error.name = 'AbortError'
      reject(error)
    }, { once: true })
  })
  await assert.rejects(
    () => requestWebResearch('http://127.0.0.1:8181', 'Search the web', { timeoutMs: 5 }),
    /timed out/,
    'a stalled research helper must degrade back to local chat instead of hanging the composer',
  )

  let parentAbortReachedFetch = false
  globalThis.fetch = async (_url, init) => new Promise((resolve, reject) => {
    init.signal.addEventListener('abort', () => {
      parentAbortReachedFetch = true
      const error = new Error('aborted by caller')
      error.name = 'AbortError'
      reject(error)
    }, { once: true })
  })
  const parentController = new AbortController()
  const parentCancelled = requestWebResearch(
    'http://127.0.0.1:8181',
    'Search the web',
    { signal: parentController.signal, timeoutMs: 1_000 },
  )
  parentController.abort()
  await assert.rejects(parentCancelled, { name: 'AbortError' })
  assert.equal(parentAbortReachedFetch, true, 'a parent chat cancellation must abort the in-flight research request')
} finally {
  globalThis.fetch = nativeFetch
}

const dashboardHookSource = await readFile(resolve(scriptDir, '../src/hooks/useDashboardData.js'), 'utf8')
const chatWorkspaceSource = await readFile(resolve(scriptDir, '../src/views/ChatWorkspace.jsx'), 'utf8')
const messageTurnSource = await readFile(resolve(scriptDir, '../src/components/chat/MessageTurn.jsx'), 'utf8')
assert.match(dashboardHookSource, /if \(webResearchEnabled\) \{[\s\S]*requestWebResearch/, 'Web Auto should let the backend classify every prompt')
assert.doesNotMatch(dashboardHookSource, /webResearchEnabled\s*&&\s*researchPlan\.needed/, 'a browser regex must not be the authoritative research gate')
assert.doesNotMatch(dashboardHookSource, /\btools\s*:/, 'Gemma Web research must not add unsupported function tools to chat requests')
assert.doesNotMatch(dashboardHookSource, /\btool_choice\s*:/, 'Web Auto must remain separate from native model tool selection')
assert.doesNotMatch(dashboardHookSource, /\bcamelid_tools\s*:/, 'Web Auto must not enter a camelid_tools loop')
assert.match(dashboardHookSource, /requirements\?\|architecture\|phases\?/, 'planning requirements, architecture, and phases must stay outside runnable-code policy')
assert.match(dashboardHookSource, /web_research_ms: webResearchMs/, 'public-web latency must be persisted separately from model TTFT and decode rate')
assert.match(dashboardHookSource, /now - firstTokenAt/, 'decode tok\/s must start at the first generated token instead of including research or TTFT')
assert.match(dashboardHookSource, /responseIsStreaming \? tokensPerSecond\(decodedTokenCount, decodeElapsedMs\) : null/, 'non-streamed or replay-shaped answers must not report browser-invented decode throughput')
assert.match(chatWorkspaceSource, /Web Auto will send linked URLs or a search query to the public web/, 'a triggering draft must disclose public-web egress before send, including on touch devices')
assert.match(messageTurnSource, /warnings\[0\]/, 'failed research should retain the backend\'s actionable warning in the reply')

console.log('Web research smoke passed')
