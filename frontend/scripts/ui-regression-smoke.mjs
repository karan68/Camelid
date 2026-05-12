#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  NEW_CHAT_SENTINEL,
  resolveSelectedConversation,
  shouldCreateConversationForSend,
} from '../src/lib/chatState.js'

const oldChat = { id: 'old-chat', title: 'Old chat', messages: [{ role: 'user', content: 'old prompt' }] }
const newerChat = { id: 'newer-chat', title: 'Newer chat', messages: [{ role: 'user', content: 'newer prompt' }] }
const conversations = [newerChat, oldChat]

assert.equal(resolveSelectedConversation(conversations, NEW_CHAT_SENTINEL), null, 'new-chat sentinel must render an empty landing, not the newest old chat')
assert.equal(resolveSelectedConversation(conversations, null), null, 'null selection must not silently fall back to an old chat')
assert.equal(resolveSelectedConversation(conversations, 'missing-chat'), null, 'missing selection must not silently fall back to an old chat')
assert.equal(resolveSelectedConversation(conversations, 'old-chat'), oldChat, 'explicit old-chat selection should still open that chat')
assert.equal(shouldCreateConversationForSend(null, NEW_CHAT_SENTINEL), true, 'sending from new-chat landing should create a fresh conversation')
assert.equal(shouldCreateConversationForSend(oldChat, NEW_CHAT_SENTINEL), true, 'the sentinel must win even if a stale selectedConversation prop exists')
assert.equal(shouldCreateConversationForSend(oldChat, 'old-chat'), false, 'sending from an explicit existing chat should append to that chat')

const readmeSource = readFileSync(new URL('../../README.md', import.meta.url), 'utf8')
assert.match(readmeSource, /docs\/assets\/camelid-readme-chat-surface-dark\.png/, 'README should use the approved dark collapsed-rail chat screenshot')
assert.doesNotMatch(readmeSource, /docs\/assets\/ui-screenshot-v2\.png/, 'README must not regress to the retired light screenshot')
assert.match(readmeSource, /product-forward while still reflecting the local-first runtime contract/i, 'README screenshot caption should preserve the local-first runtime contract')

const chatWorkspaceSource = readFileSync(new URL('../src/views/ChatWorkspace.jsx', import.meta.url), 'utf8')
const dashboardHookSource = readFileSync(new URL('../src/hooks/useDashboardData.js', import.meta.url), 'utf8')
const apiViewSource = readFileSync(new URL('../src/views/ApiView.jsx', import.meta.url), 'utf8')
const systemViewSource = readFileSync(new URL('../src/views/SystemView.jsx', import.meta.url), 'utf8')
const analyticsViewSource = readFileSync(new URL('../src/views/AnalyticsView.jsx', import.meta.url), 'utf8')
const capabilitiesSource = readFileSync(new URL('../src/lib/capabilities.js', import.meta.url), 'utf8')
const streamParserSource = readFileSync(new URL('../src/lib/chatCompletionStream.js', import.meta.url), 'utf8')
const visibleUiSources = [
  '../src/views/ChatWorkspace.jsx',
  '../src/views/ApiView.jsx',
  '../src/views/SystemView.jsx',
  '../src/views/ModelsView.jsx',
  '../src/hooks/useDashboardData.js',
].map((path) => [path, readFileSync(new URL(path, import.meta.url), 'utf8')])
assert.match(chatWorkspaceSource, /pending is-streaming/, 'pending assistant row should use the same streaming Pac-Man state as live token rows')
assert.match(chatWorkspaceSource, /splitFenceInfo/, 'streaming/incomplete fenced code blocks should be parsed as code instead of prose')
assert.match(chatWorkspaceSource, /pushCodeBlock/, 'code block rendering should stay centralized for complete and incomplete fences')
assert.match(chatWorkspaceSource, /streaming=\{assistantStreaming\}/, 'assistant markdown should know when an assistant row is still streaming')
assert.match(chatWorkspaceSource, /\$\{assistantStreaming \? 'is-streaming' : ''\}/, 'only assistant rows that are actively streaming should receive the animated streaming class')
assert.doesNotMatch(chatWorkspaceSource, /\$\{message\.streaming \? 'is-streaming' : ''\}/, 'raw message.streaming should not keep completed/non-assistant rows visually active')
assert.match(chatWorkspaceSource, /incomplete:\s*incompleteFence,\s*streaming/, 'unclosed streaming fences should reach the code-card renderer as active incomplete code')
assert.match(chatWorkspaceSource, /aria-busy=\{stillGenerating \? 'true' : undefined\}/, 'incomplete streaming code cards should expose busy state')
assert.match(chatWorkspaceSource, /hasOpenCodeFence/, 'streaming rows should detect open fenced code so the active state can call that out')
assert.match(chatWorkspaceSource, /isOpenStreamingCode\s*=\s*assistantStreaming\s*&&\s*hasOpenCodeFence\(messageContent\)/, 'open-code status should only be active on rows that are still streaming')
assert.match(chatWorkspaceSource, /data-streaming-code-state=\{isOpenStreamingCode \? 'open' : undefined\}/, 'only open streaming code rows should expose the active code-state marker')
assert.match(chatWorkspaceSource, /data-live-status="active"/, 'visible still-generating status badges should expose an active marker only when rendered')
assert.match(chatWorkspaceSource, /ACTIVE_STREAMING_LABEL\s*=\s*'Still generating — response is active'/, 'live assistant rows should keep an explicit still-generating status while the backend is active')
assert.match(chatWorkspaceSource, /OPEN_CODE_STREAMING_LABEL\s*=\s*'Still generating — code block is still open'/, 'streaming open code fences should visibly say the code block is still incomplete')
assert.match(chatWorkspaceSource, /assistantStreaming && <StreamingStatus/, 'token-streaming assistant rows should render the active status badge before streamed content')
assert.match(chatWorkspaceSource, /FIRST_TOKEN_STREAMING_LABEL\s*=\s*'Still generating; waiting for the first token'/, 'pre-token pending rows should visibly say the backend is still generating')
assert.match(chatWorkspaceSource, /aria-busy=\{assistantStreaming \? 'true' : undefined\}/, 'streaming assistant rows should expose row-level busy state while text is incomplete')
assert.match(chatWorkspaceSource, /data-streaming-state=\{assistantStreaming \? 'active' : undefined\}/, 'streaming assistant rows should expose an active state marker for regression coverage')
assert.match(chatWorkspaceSource, /pending is-streaming" aria-busy="true" data-streaming-state="active"/, 'pre-token pending rows should be active while the backend is thinking')
assert.match(chatWorkspaceSource, /lastVisibleMessageIsUser[\s\S]*awaitingAssistant[\s\S]*sending && !hasStreamingAssistant/, 'a sent user row should keep showing an awaiting assistant indicator until the streaming row is visible')
assert.match(chatWorkspaceSource, /hasStreamingAssistant[\s\S]*generationActive/, 'a persisted streaming row should keep the UI active even if the send call state changes')
assert.match(chatWorkspaceSource, /assistantStreaming && <StreamingStatus[^\n]*tail/, 'streaming assistant rows should keep a visible active status after incomplete streamed content too')
assert.match(chatWorkspaceSource, /CODE_CARD_STREAMING_LABEL\s*=\s*'Still generating — code block incomplete'/, 'incomplete streaming code blocks should visibly say the code is still incomplete')
assert.match(chatWorkspaceSource, /data-code-streaming-state=\{stillGenerating \? 'open' : undefined\}/, 'open streaming code fences should expose an active code state marker')
assert.match(chatWorkspaceSource, /message-code-card-status[^>]*aria-live="polite"[^>]*data-live-status="active"[^>]*>\{CODE_CARD_STREAMING_LABEL\}</, 'incomplete streaming code blocks should show a live active still-generating badge')
assert.match(dashboardHookSource, /readStreamingChatCompletion\(response/, 'dashboard chat send should use the centralized stream parser')
assert.match(streamParserSource, /function defaultEstimateTokenCount/, 'central stream parser should keep a JSON fallback token estimator')
assert.match(streamParserSource, /function readSseDataLines/, 'central stream parser should isolate SSE data-line handling')
assert.match(streamParserSource, /export function extractSseEvents/, 'stream parser should keep SSE boundary handling centralized')
assert.match(streamParserSource, /replace\(/, 'stream parser should normalize line endings before splitting SSE events')
assert.match(streamParserSource, /split\('\\n\\n'\)/, 'stream parser should split normalized SSE events on blank lines for partial rendering')
assert.match(dashboardHookSource, /finish_reason:\s*'error',[\s\S]*streaming:\s*false/, 'failed generations should clear streaming state instead of leaving active pellets/status forever')
assert.match(apiViewSource, /Selected exact-row evidence/, 'API support view should show selected exact-row evidence instead of a broad validated-target claim')
assert.match(apiViewSource, /selectedExactRowReady/, 'API endpoint readiness should only turn green when runtime readiness and the selected exact compatibility row both match')
assert.match(apiViewSource, /selectedRuntimeMatches/, 'API endpoint readiness should require active_model_id to match the selected model')
assert.match(apiViewSource, /readinessPillCopy/, 'API endpoint status copy should come from the exact-row readiness gate, not generation_ready alone')
assert.match(apiViewSource, /chatCompletionsCopy/, 'API chat-completions copy should stay gated unless selected exact-row evidence and runtime readiness both match')
assert.match(apiViewSource, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl example should fail closed until exact-row evidence and runtime readiness match')
assert.match(apiViewSource, /selectedCompatibilityTarget\.frontend_readiness_gate/, 'API support view should surface the selected row readiness gate verbatim from /api/capabilities')
assert.match(apiViewSource, /selectedCompatibilityTarget\.support_scope/, 'API support view should surface exact-row support scope instead of inferring a broader claim')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_bucket/, 'API support view should surface exact-row latest checked bucket evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_output/, 'API support view should surface exact-row latest output evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.full_support_status/, 'API support view should show the exact row full-support status boundary')
assert.match(apiViewSource, /selectedCompatibilityTarget\.full_support_blockers/, 'API support view should show remaining exact-row blockers before any broader support claim')
assert.match(apiViewSource, /displayCapabilityCopy\(selectedCompatibilityTarget\.evidence\)/, 'API support view should sanitize and display exact-row evidence copy')
assert.match(capabilitiesSource, /function displayCapabilityId/, 'capability ids should be display-normalized before support/API UI rendering')
assert.match(capabilitiesSource, /function displayCapabilityCopy/, 'backend capability copy should be display-normalized before support/API UI rendering')
assert.match(apiViewSource, /displayCapabilityId\(feature\.id\)/, 'API view should not render raw provider-scoped API feature ids')
assert.match(systemViewSource, /displayCapabilityId\(feature\.id\)/, 'System view should not render raw provider-scoped API feature ids')
assert.match(analyticsViewSource, /displayCapabilityId\(feature\.id\)/, 'Analytics view should not render raw provider-scoped API feature ids')
for (const [path, source] of visibleUiSources) {
  assert.doesNotMatch(source, /\b(OpenAI|ChatGPT|Claude|Gemini)\b/, `${path} visible copy should not mention competitor brands`)
}
assert.doesNotMatch(chatWorkspaceSource, /max[-_\s]?tokens?|token\s+limit/i, 'Chat UI should not expose a visible max-token picker or cap')

const componentCss = readFileSync(new URL('../src/styles/components.css', import.meta.url), 'utf8')
assert.match(componentCss, /assistant\.is-streaming::before\s*{[^}]*camelid-pacman-chomp/s, 'Pac-Man should chomp while streaming')
assert.match(componentCss, /assistant\.is-streaming::after\s*{[^}]*camelid-pellets-feed/s, 'pellets should only appear on streaming assistant rows')
assert.match(componentCss, /assistant:not\(\.is-streaming\)::before\s*{[^}]*animation:\s*none/s, 'completed assistant rows should explicitly keep Pac-Man non-animated')
assert.match(componentCss, /assistant:not\(\.is-streaming\)::after\s*{[^}]*content:\s*none[^}]*animation:\s*none/s, 'completed assistant rows should explicitly suppress pellet pseudo-content and animation')
assert.match(componentCss, /\.message-live-status\s*{[^}]*border-radius:\s*999px/s, 'streaming assistant rows should have a compact visible active badge')
assert.match(componentCss, /\.message-live-status-compact\s*{[^}]*margin-top:\s*0[^}]*margin-bottom:\s*12px/s, 'active badges should sit above streamed content instead of hiding below partial code')
assert.match(componentCss, /\.message-live-status-tail\s*{[^}]*margin-top:\s*12px[^}]*margin-bottom:\s*0/s, 'active badges should also sit after streamed content so long partial code does not look stopped')
assert.match(componentCss, /\.message-code-card\.is-generating\s*{/, 'incomplete streaming code cards should have an active visual treatment')
assert.match(componentCss, /\.message-code-card\.is-generating figcaption\s*{[^}]*position:\s*sticky[^}]*top:\s*0/s, 'incomplete streaming code cards should keep their generating label anchored while code grows')
assert.match(componentCss, /\.message-code-card-status\s*{[^}]*display:\s*inline-flex[^}]*gap:\s*6px/s, 'incomplete code badges should be compact and visibly active')
assert.match(componentCss, /\.message-code-card-status::before\s*{[^}]*animation:\s*camelid-live-dot-pulse/s, 'incomplete code badges should pulse only when the badge is rendered')
const pacmanRule = componentCss.match(/\/\* Tiny Pac-Man assistant marker[\s\S]*?\.message-row-gemini\.assistant\.is-streaming::before/s)?.[0] || ''
assert.match(pacmanRule, /--assistant-pacman-size:\s*12px/, 'Pac-Man should stay small')
assert.match(pacmanRule, /width:\s*var\(--assistant-pacman-size\)/, 'Pac-Man width should stay tied to the small game marker size')
assert.match(pacmanRule, /height:\s*var\(--assistant-pacman-size\)/, 'Pac-Man height should stay tied to the small game marker size')
assert.match(pacmanRule, /transform:\s*none/, 'Pac-Man should not bob or float')
assert.match(pacmanRule, /filter:\s*none/, 'Pac-Man should not look like it is floating above the row')
assert.match(pacmanRule, /position:\s*absolute/, 'Pac-Man should stay anchored rather than float in the text flow')
const streamingPelletRule = componentCss.match(/\.message-row-gemini\.assistant\.is-streaming::after\s*{[\s\S]*?\n}/)?.[0] || ''
assert.match(streamingPelletRule, /transform:\s*none/, 'streaming pellets should not bob or float')
assert.match(streamingPelletRule, /filter:\s*none/, 'streaming pellets should not use a floating glow/drop-shadow')
assert.match(streamingPelletRule, /width:\s*24px/, 'streaming pellets should stay small and game-like')
const pelletKeyframes = componentCss.match(/@keyframes camelid-pellets-feed\s*{[\s\S]*?\n}/)?.[0] || ''
assert.doesNotMatch(pelletKeyframes, /translateY|scale[XY]?\(/, 'pellet animation should stay game-steady without bobbing or scaling')
const pacmanAndPellets = `${pacmanRule}\n${streamingPelletRule}\n${pelletKeyframes}`
assert.doesNotMatch(componentCss, /camelid-pacman-bob/, 'Pac-Man should stay game-steady instead of bobbing')
assert.doesNotMatch(pacmanAndPellets, /translateY|translate3d|scale[XY]?\(/, 'Pac-Man and pellet styling should not bob, float, or scale')
assert.doesNotMatch(componentCss, /assistant::after\s*{/, 'completed assistant rows must not keep pellet animation')

console.log('UI regression smoke passed')
