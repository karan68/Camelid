#!/usr/bin/env node
/* UI regression smoke — re-baselined in Phase 2 pre-work against the shipped
   Phase 1 reality (Evidence Chip, dark-first tokens, post-redesign chat stack).

   Heritage: every assertion from the pre-rebaseline script was either ported
   (verbatim where the source still matches, re-pointed where the code moved
   to MessageTurn.jsx / lib/markdown.jsx / chat.css) or explicitly retired —
   the retirement list with reasons lives in the re-baseline commit message.
   From that commit onward this smoke is part of the standing I6 gate set. */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  NEW_CHAT_SENTINEL,
  resolveSelectedConversation,
  shouldCreateConversationForSend,
} from '../src/lib/chatState.js'
import { normalizeStoredConversations } from '../src/lib/conversationStorage.js'
import { conversationToJson, conversationToMarkdown } from '../src/lib/conversationExport.js'

/* ---- Behavioral asserts: conversation selection + stored-stream recovery ---- */
const oldChat = { id: 'old-chat', title: 'Old chat', messages: [{ role: 'user', content: 'old prompt' }] }
const newerChat = { id: 'newer-chat', title: 'Newer chat', messages: [{ role: 'user', content: 'newer prompt' }] }
const conversations = [newerChat, oldChat]

assert.equal(resolveSelectedConversation(conversations, NEW_CHAT_SENTINEL), null, 'new-chat sentinel must render an empty landing, not the newest old chat')
assert.equal(resolveSelectedConversation(conversations, null), newerChat, 'null selection should recover to the newest available chat so the main pane does not blank during streaming')
assert.equal(resolveSelectedConversation(conversations, 'missing-chat'), newerChat, 'missing selection should recover to the newest available chat so streaming stays attached to a visible thread')
assert.equal(resolveSelectedConversation(conversations, 'old-chat'), oldChat, 'explicit old-chat selection should still open that chat')
assert.equal(shouldCreateConversationForSend(null, NEW_CHAT_SENTINEL), true, 'sending from new-chat landing should create a fresh conversation')
assert.equal(shouldCreateConversationForSend(oldChat, NEW_CHAT_SENTINEL), true, 'the sentinel must win even if a stale selectedConversation prop exists')
assert.equal(shouldCreateConversationForSend(oldChat, 'old-chat'), false, 'sending from an explicit existing chat should append to that chat')

const revivedInterruptedChat = normalizeStoredConversations([{ id: 'stale-chat', messages: [{ id: 'stale-assistant', role: 'assistant', content: '', streaming: true, streaming_phase: 'streaming' }] }], { clearStaleStreaming: true })[0]
assert.equal(revivedInterruptedChat.messages[0].streaming, false, 'reloaded interrupted streams should not claim the backend is still generating')
assert.equal(revivedInterruptedChat.messages[0].streaming_phase, null, 'reloaded interrupted streams should clear live generation phase')
assert.equal(revivedInterruptedChat.messages[0].finish_reason, 'interrupted', 'reloaded interrupted streams should be marked as interrupted')
assert.equal(revivedInterruptedChat.messages[0].content, '(generation interrupted)', 'blank reloaded interrupted streams should render safely')
const liveStreamingChat = normalizeStoredConversations([{ id: 'live-chat', messages: [{ id: 'live-assistant', role: 'assistant', content: 'partial', streaming: true, streaming_phase: 'streaming' }] }])[0]
assert.equal(liveStreamingChat.messages[0].streaming, true, 'live in-memory stream normalization should preserve active generation state')

/* ---- Conversation export must be path-free by construction (I7) ---- */
const sneakyConversation = {
  id: 'conv-1',
  title: 'Export test',
  model_id: 'llama32_3b_instruct_q8_0',
  model_path: '/Volumes/Untitled/models/secret.gguf',
  messages: [{
    id: 'm1', role: 'assistant', content: 'hello', model_id: 'llama32_3b_instruct_q8_0',
    model_path: '/Volumes/Untitled/models/secret.gguf',
    camelid: { backend_path: '/private/tmp/x.gguf' },
    support_row: { id: 'llama32_3b_instruct_q8_0', status: 'supported_exact_row_smoke', supported: true, manifest_path: '/Volumes/ExampleHome/qa/manifest.json' },
    usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
    usage_source: 'backend',
  }],
}
for (const exported of [conversationToJson(sneakyConversation), conversationToMarkdown(sneakyConversation)]) {
  assert.doesNotMatch(exported, /model_path|backend_path|manifest_path|\/Volumes\/|\/private\/tmp|\/Users\//, 'exports must never include filesystem paths — whitelisted fields only')
}
assert.match(conversationToJson(sneakyConversation), /telemetry, not support evidence|telemetry_note/, 'exports must carry the telemetry-not-evidence note')
assert.match(conversationToMarkdown(sneakyConversation), /telemetry, not support evidence/, 'markdown exports must label telemetry as not support evidence')

/* ---- Sources ---- */
const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8')
const readmeSource = read('../../README.md')
const chatWorkspaceSource = read('../src/views/ChatWorkspace.jsx')
const messageTurnSource = read('../src/components/chat/MessageTurn.jsx')
const markdownSource = read('../src/lib/markdown.jsx')
const dashboardHookSource = read('../src/hooks/useDashboardData.js')
const loadedModelDisplaySource = read('../src/lib/loadedModelDisplay.js')
const apiViewSource = read('../src/views/ApiView.jsx')
const systemViewSource = read('../src/views/SystemView.jsx')
const modelsViewSource = read('../src/views/ModelsView.jsx')
const topBarSource = read('../src/components/TopBar.jsx')
const analyticsViewSource = read('../src/views/AnalyticsView.jsx')
const capabilitiesSource = read('../src/lib/capabilities.js')
const streamParserSource = read('../src/lib/chatCompletionStream.js')
const evidenceChipSource = read('../src/components/ui/EvidenceChip.jsx')
const modelInspectorSource = read('../src/components/models/ModelInspector.jsx')
const compatibilityViewSource = read('../src/views/CompatibilityView.jsx')
const apiWorkbenchSource = read('../src/components/api/ApiWorkbench.jsx')
const telemetryViewSource = read('../src/views/TelemetryView.jsx')
const telemetryLogSource = read('../src/lib/telemetryLog.js')
const appSource = read('../src/App.jsx')
const tokenizerPlaygroundSource = read('../src/components/models/TokenizerPlayground.jsx')
const evidenceStatusSource = read('../src/lib/evidenceStatus.js')
const useThemeSource = read('../src/hooks/useTheme.js')
const mainSource = read('../src/main.jsx')
const tokensCss = read('../src/styles/tokens.css')
const evidenceCss = read('../src/styles/evidence.css')
const chatCss = read('../src/styles/chat.css')
const uiCss = read('../src/styles/ui.css')
const statusSheets = ['../src/styles/ui.css', '../src/styles/shell.css', '../src/styles/chat.css', '../src/styles/views.css', '../src/styles/cluster.css', '../src/styles/observatory.css']
  .map((path) => [path, read(path)])

/* ---- README product surface ---- */
assert.match(readmeSource, /docs\/assets\/camelid-readme-chat-surface-dark\.png/, 'README should use the approved dark collapsed-rail chat screenshot')
assert.doesNotMatch(readmeSource, /assets\/camelid-banner\.png/, 'README should not lead with the disliked first banner image')
assert.doesNotMatch(readmeSource, /docs\/assets\/ui-screenshot-v2\.png/, 'README must not regress to the retired light screenshot')

/* ---- Chat workspace ---- */
assert.match(chatWorkspaceSource, /lastVisibleMessageIsUser[\s\S]*awaitingAssistant[\s\S]*generationActive && !hasStreamingAssistantContent/, 'a sent user row should keep showing an awaiting assistant indicator until streamed assistant content is visible')
assert.match(chatWorkspaceSource, /hasStreamingAssistant[\s\S]*generationActive/, 'a persisted streaming row should keep the UI active even if the send call state changes')
assert.match(chatWorkspaceSource, /cxcomposer__status/, 'the composer should keep the consolidated runtime/support status line')
assert.match(chatWorkspaceSource, /<EvidenceChip/, 'the composer support claim should render through the Evidence Chip, not an ad-hoc badge')
assert.match(chatWorkspaceSource, /selectedChatGate\.contractSupported \? 'supported'/, 'the composer Evidence Chip must take its supported state only from the shared chat gate')
assert.doesNotMatch(chatWorkspaceSource, /max[-_\s]?tokens?|token\s+limit/i, 'Chat UI should not expose a visible max-token picker or cap')

/* ---- Message rendering (moved from pre-redesign ChatWorkspace to MessageTurn/markdown) ---- */
assert.match(messageTurnSource, /aria-busy=\{assistantStreaming \? 'true' : undefined\}/, 'streaming assistant rows should expose row-level busy state while text is incomplete')
assert.match(messageTurnSource, /data-streaming-state=\{assistantStreaming \? 'active' : undefined\}/, 'streaming assistant rows should expose an active state marker for regression coverage')
assert.match(messageTurnSource, /\$\{assistantStreaming \? 'is-streaming' : ''\}/, 'only assistant rows that are actively streaming should receive the animated streaming class')
assert.doesNotMatch(messageTurnSource, /\$\{message\.streaming \? 'is-streaming' : ''\}/, 'raw message.streaming should not keep completed/non-assistant rows visually active')
assert.match(messageTurnSource, /streaming=\{assistantStreaming\}/, 'assistant markdown should know when an assistant row is still streaming')
assert.match(markdownSource, /splitFenceInfo/, 'streaming/incomplete fenced code blocks should be parsed as code instead of prose')
assert.match(markdownSource, /pushCodeBlock/, 'code block rendering should stay centralized for complete and incomplete fences')
assert.match(markdownSource, /CODE_CARD_STREAMING_LABEL\s*=\s*'Still generating — code block incomplete'/, 'incomplete streaming code blocks should visibly say the code is still incomplete')
assert.match(markdownSource, /data-code-streaming-state=\{stillGenerating \? 'open' : undefined\}/, 'open streaming code fences should expose an active code state marker')
assert.match(markdownSource, /message-code-card-status[^>]*aria-live="polite"[^>]*data-live-status="active"[^>]*>\{CODE_CARD_STREAMING_LABEL\}</, 'incomplete streaming code blocks should show a live active still-generating badge')
assert.doesNotMatch(markdownSource, /dangerouslySetInnerHTML/, 'model output must never reach the DOM through dangerouslySetInnerHTML')
assert.doesNotMatch(messageTurnSource, /dangerouslySetInnerHTML/, 'message rows must never use dangerouslySetInnerHTML')

/* ---- Dashboard data hook ---- */
assert.match(dashboardHookSource, /Include inline <style> and inline <script>/, 'HTML code prompts should ask for inline CSS and JS, not an unfinished fragment')
assert.match(dashboardHookSource, /max_tokens:\s*localChatMaxTokens\(history\)/, 'local chat sends should choose the token budget from the prompt policy')
assert.match(dashboardHookSource, /getRuntimeRequestModelId\(selectedModel, runtime, selectedModelId\)/, 'chat sends should use the backend active runtime model id when a browser alias is selected')
assert.doesNotMatch(dashboardHookSource, /Camelid streamed the local reply\./, 'successful streams should not show a noisy demo-breaking toast')
assert.match(dashboardHookSource, /readStreamingChatCompletion\(response/, 'dashboard chat send should use the centralized stream parser')
assert.match(dashboardHookSource, /finish_reason:\s*requestWasAborted\s*\?\s*'interrupted'\s*:\s*'error',[\s\S]*streaming:\s*false/, 'failed or interrupted generations should clear streaming state instead of leaving active pellets/status forever')
assert.match(dashboardHookSource, /const conversations = localConversations\.length \? localConversations : dashboard\?\.conversations \|\| \[\]/, 'main chat should resolve selectedConversation from live local conversation state before stale dashboard snapshots')
assert.match(dashboardHookSource, /currentLocalConversations\.some\(\(conversation\) => conversation\.id === current\)/, 'dashboard refresh should validate selected conversation against the same current local conversation snapshot it renders')
assert.match(dashboardHookSource, /const selectedConversationIdRef = useRef\(selectedConversationId\)/, 'conversation selection should keep an immediate ref so background refreshes do not lose the active thread between state commits')
assert.match(dashboardHookSource, /selectedConversationIdRef\.current = next[\s\S]*setSelectedConversationIdState\(next\)/, 'conversation selection updates should write the ref immediately before the async state commit')
assert.match(dashboardHookSource, /activeModelChatGate\?\.chatUnlocked && current !== activeModel\.id/, 'browser-selected model should snap back to the backend active model only through the shared exact-row chat gate')
assert.match(dashboardHookSource, /modelRuntimeIdMatches/, 'dashboard model merge should treat runtime_model_name as an active_model_id alias instead of losing readiness for imported exact rows')
assert.match(dashboardHookSource, /resolveLoadedModelDisplayName/, 'dashboard model merge should rewrite backend-generated active ids to the exact 3B display row only from exact GGUF filename plus Q8_0 metadata')
assert.match(loadedModelDisplaySource, /ggufFileTypeValueFromLabel[\s\S]*quantLabelFromGgufFileType[\s\S]*LLAMA32_3B_ACCEPTANCE_FILENAME[\s\S]*normalizeQuantLabel\(quantLabel\) === 'Q8_0'/, 'the 3B display alias must stay exact-row and decoded Q8_0/file_type 7 gated rather than broad-family')
assert.match(dashboardHookSource, /localRecordMatchesBackendId/, 'dashboard model merge should de-duplicate backend model rows against saved browser records by id or runtime_model_name')
assert.match(dashboardHookSource, /const id = localRecord\?\.id \|\| item\.id/, 'backend model merges should preserve the browser row id while keeping the backend runtime id as runtime_model_name')
assert.match(dashboardHookSource, /const conversation = await ensureConversation\(\)[\s\S]*?setSelectedConversationId\(conversation\.id\)[\s\S]*?fetch\(`\$\{normalizedApiBase\}\/v1\/chat\/completions`/, 'fresh-chat sends must select the real conversation before streaming starts so the main pane updates with sidebar previews')
assert.match(dashboardHookSource, /applyLocalChatPolicy\(history\)/, 'code/html prompts should use the local code-first request policy')
assert.match(dashboardHookSource, /CODE_FIRST_SYSTEM_PROMPT/, 'frontend should keep a code-first system prompt for code/html local chat requests')
assert.match(dashboardHookSource, /begin immediately with complete runnable code/, 'code-first prompt should suppress slow prose preambles before code and ask for complete output')
assert.match(dashboardHookSource, /Start exactly with ```html then <!doctype html>/, 'HTML code prompts should request visible code at the beginning of the stream')
assert.match(dashboardHookSource, /ONE self-contained file/, 'HTML code prompts should ask for one complete file, not separated assets')
assert.match(dashboardHookSource, /For Python, start exactly with ```python/, 'Python code prompts should get a Python-specific complete-script instruction')
assert.match(dashboardHookSource, /prefer tkinter from the standard library over pygame/, 'Python game prompts should prefer compact standard-library demos over sprawling dependency-heavy pygame output')
assert.match(dashboardHookSource, /complete runnable event loop/, 'Python game prompts should ask for runnable game logic, not a sketch')
assert.match(dashboardHookSource, /python\|py\|pygame\|game\|pacman\|pacmac/, 'code-first detection should catch Python game demos and the pacmac typo')
assert.match(dashboardHookSource, /Never use external files or script src/, 'HTML code prompts should prevent unusable external script references in demos')

/* ---- Stream parser ---- */
assert.match(streamParserSource, /function defaultEstimateTokenCount/, 'central stream parser should keep a JSON fallback token estimator')
assert.match(streamParserSource, /function readSseDataLines/, 'central stream parser should isolate SSE data-line handling')
assert.match(streamParserSource, /export function extractSseEvents/, 'stream parser should keep SSE boundary handling centralized')
assert.match(streamParserSource, /replace\(/, 'stream parser should normalize line endings before splitting SSE events')
assert.match(streamParserSource, /split\('\\n\\n'\)/, 'stream parser should split normalized SSE events on blank lines for partial rendering')

/* ---- API view ---- */
assert.match(apiViewSource, /Selected exact-row evidence/, 'API support view should show selected exact-row evidence instead of a broad validated-target claim')
assert.match(apiViewSource, /selectedChatGate\s*=\s*getChatGateState\(capabilities, selectedModel, runtime\)/, 'API endpoint readiness should use the shared exact-row chat gate')
assert.match(apiViewSource, /selectedExactRowReady\s*=\s*selectedChatGate\.chatUnlocked/, 'API endpoint readiness should stay aligned with Chat/System exact-row chat unlocks')
assert.match(apiViewSource, /selectedRuntimeMatches/, 'API endpoint readiness should require active_model_id to match the selected model')
assert.match(apiViewSource, /readinessPillCopy/, 'API endpoint status copy should come from the exact-row readiness gate, not generation_ready alone')
assert.match(apiViewSource, /chatCompletionsCopy/, 'API chat-completions copy should stay gated unless selected exact-row evidence and runtime readiness both match')
assert.match(apiViewSource, /Blocked for UX chat until selected exact row evidence and runtime readiness both match/, 'API curl example should fail closed until exact-row evidence and runtime readiness match')
assert.match(apiViewSource, /selectedCompatibilityTarget\.frontend_readiness_gate/, 'API support view should surface the selected row readiness gate verbatim from /api/capabilities')
assert.match(apiViewSource, /selectedCompatibilityTarget\.support_scope/, 'API support view should surface exact-row support scope instead of inferring a broader claim')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_bucket/, 'API support view should surface exact-row latest checked bucket evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.latest_checked_output/, 'API support view should surface exact-row latest output evidence')
assert.match(apiViewSource, /selectedCompatibilityTarget\.full_support_status/, 'API support view should show the exact row full-support status boundary')
assert.match(apiViewSource, /exactRowSupportLanes\(selectedCompatibilityTarget, apiFeatures\)/, 'API support view should show template/Jinja, checked-context, and throughput readiness lanes for the selected exact row')
assert.match(apiViewSource, /rowSupportBoundaryCopy\(selectedCompatibilityTarget, apiFeatures\)/, 'API support view should filter resolved template/Jinja and throughput blockers out of the remaining support boundary')
assert.match(apiViewSource, /rowSupportNextStepCopy\(target, apiFeatures\)/, 'API support view should filter resolved template/Jinja and throughput blockers out of row next-step copy')
assert.match(capabilitiesSource, /function frontendSupportContractCopy/, 'frontend support contract copy should filter resolved template/Jinja and throughput caveats for current supported rows')
assert.match(capabilitiesSource, /Production-throughput readiness is green/, 'capability helpers should describe production-throughput as a green exact-row readiness lane when perf evidence is supported')
assert.match(apiViewSource, /function summarizeExactRowField/, 'API support view should summarize quant and family evidence from exact compatibility rows')
assert.match(apiViewSource, /Exact-row quant evidence/, 'API support view should label quant evidence as exact-row scoped')
assert.match(apiViewSource, /Exact-row family evidence/, 'API support view should label family evidence as exact-row scoped')
assert.match(apiViewSource, /broad quant lists do not unlock chat/, 'API support view should not promote broad quant lists into chat readiness')
assert.match(apiViewSource, /row-scoped family\/quant evidence/, 'API endpoint summary should describe family and quant evidence as row-scoped')
assert.doesNotMatch(apiViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|summarizeCapabilityItems/, 'API support view should not render non-row capability lists as support evidence')
assert.match(apiViewSource, /No exact compatibility row matched this selected model/, 'API selected model contract should fail closed instead of displaying family or saved-path guesses')
assert.match(apiViewSource, /displayCapabilityCopy\(selectedCompatibilityTarget\.evidence\)/, 'API support view should sanitize and display exact-row evidence copy')
assert.match(capabilitiesSource, /function displayCapabilityId/, 'capability ids should be display-normalized before support/API UI rendering')
assert.match(capabilitiesSource, /function displayCapabilityCopy/, 'backend capability copy should be display-normalized before support/API UI rendering')
assert.match(apiViewSource, /displayCapabilityId\(feature\.id\)/, 'API view should not render raw provider-scoped API feature ids')
assert.match(apiViewSource, /getRuntimeRequestModelId\(selectedModel, runtime, '<loaded-model-id>'\)/, 'API curl examples should use the loaded backend model id for alias-selected exact rows')
assert.match(apiViewSource, /<EvidenceChip/, 'API contract rows should render their status claims through the Evidence Chip')

/* ---- System view ---- */
assert.match(systemViewSource, /Selected exact-row evidence/, 'System support view should show selected exact-row evidence instead of broad quant or family capability lists')
assert.match(systemViewSource, /Exact-row quant evidence/, 'System support view should scope quant evidence to compatibility rows')
assert.match(systemViewSource, /Exact-row family evidence/, 'System support view should scope family evidence to compatibility rows')
assert.doesNotMatch(systemViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|summarizeCapabilityItems/, 'System support view should not render non-row capability lists as support evidence')
assert.match(systemViewSource, /displayCapabilityId\(feature\.id\)/, 'System view should not render raw provider-scoped API feature ids')
assert.match(systemViewSource, /getRuntimeRequestModelId\(selectedModel, runtime, '<loaded-model-id>'\)/, 'System curl examples should use the loaded backend model id for alias-selected exact rows')
assert.match(systemViewSource, /<EvidenceChip/, 'System contract rows should render their status claims through the Evidence Chip')

/* ---- Models view ---- */
assert.match(modelsViewSource, /Exact-row quant evidence/, 'Models support cards should display row-scoped quant evidence')
assert.match(modelsViewSource, /Exact-row support/, 'Models support cards should display exact-row support boundaries')
assert.match(modelsViewSource, /modelRuntimeIdMatches\(model, runtime\)/, 'Models runtime/readiness surfaces should accept backend runtime_model_name aliases for exact-row local imports')
assert.match(modelsViewSource, /modelRuntimeIdMatches\(selectedLocalModel, runtime\)/, 'Models next-chat copy should not fall back to blocked when the selected exact row uses a browser id plus backend runtime_model_name')
assert.match(modelsViewSource, /compatibilityHintMatchesExactTarget\(capabilities, model, target\)/, 'Models tracked-row matching must require an exact compatibility hint, not a quant-mismatch target id')
assert.match(modelsViewSource, /matchedChatGate\s*=\s*matchedModel \? getChatGateState\(capabilities, matchedModel, runtime\) : null/, 'Models tracked exact-row cards should use the shared chat gate instead of stale browser readiness')
assert.match(modelsViewSource, /chatUnlocked\s*=\s*Boolean\(matchedChatGate\?\.chatUnlocked\)/, 'Models tracked exact-row cards should require loaded_now, generation_ready, active_model_id, and exact support before claiming chat unlock')
assert.match(modelsViewSource, /matchesLlama32ThreeBTarget\(model, capabilities\)/, 'The 3B acceptance target should only hide when an exact 3B Q8_0 row is present')
assert.match(modelsViewSource, /LLAMA32_3B_ACCEPTANCE_FILENAME[\s\S]*hasExactLlama32ThreeBArtifact[\s\S]*compatibilityHintMatchesExactTarget/, 'The 3B acceptance target fallback must require exact GGUF artifact identity instead of a broad 3B Instruct Q8 label')
assert.doesNotMatch(modelsViewSource, /findCompatibilityHint\(capabilities, model\)\?\.target\?\.id === target\.id/, 'Models tracked-row matching must not treat quant-mismatch hints as exact-row matches')
assert.match(modelsViewSource, /rowSupportNextStepCopy\(target, apiFeatures\)/, 'Models current-row cards should filter resolved template/Jinja and throughput blockers out of next-step copy')
assert.match(modelsViewSource, /template\/Jinja, checked context, and production-throughput shown as row-scoped readiness lanes instead of repeated generic caveats/, 'Models support section should state that template/Jinja, checked-context, and production-throughput are row-scoped lanes, not generic caveats')
assert.match(modelsViewSource, /Catalog quant:/, 'Catalog cards may show catalog quant labels without promoting them to support')
assert.doesNotMatch(modelsViewSource, /supported_quantization|planned_quantization|supported_model_families|planned_model_families|getQuantCapability|quantCapabilityLabel|quantCapabilityCopy/, 'Models view should not render broad quant/family capability lists as support evidence')
assert.match(modelsViewSource, /<EvidenceChip/, 'Models tracked-row status claims should render through the Evidence Chip')

/* ---- Model management (Phase 3) ---- */
assert.match(modelsViewSource, /ModelCardEvidence/, 'local model cards must resolve their claim through the card evidence chip')
assert.match(modelsViewSource, /no exact supported row/, 'unmatched local models must show the calm no-exact-row state, not an error')
assert.match(modelsViewSource, /view the compatibility ledger/, 'unmatched local models must link to the compatibility view')
assert.match(modelInspectorSource, /not support evidence/, 'the model inspector must label its contents as descriptive, not support evidence')
assert.doesNotMatch(modelInspectorSource, /getChatGateState|isCompatibilitySupportedForModel|findCompatibilityHint/, 'the inspector renders metadata; it must never compute or imply gate state')
assert.match(modelInspectorSource, /items\]|items…/, 'huge GGUF arrays must be summarized, not dumped')
assert.match(tokenizerPlaygroundSource, /does not widen generation support/, 'the tokenizer playground must say its output is not generation-support evidence')
assert.match(tokenizerPlaygroundSource, /tokenizer_encode_decode/, 'the playground chip must cite the exact contract feature row')

/* ---- Observatory lifecycle (Phase 6.1 defect guards) ---- */
const inferenceTelemetryHookSource = read('../src/hooks/useInferenceTelemetry.js')
assert.match(inferenceTelemetryHookSource, /const sharedStore = createInferenceTelemetryStore\(\)/, 'the inference telemetry store must be a shared app-lifetime singleton, not per-mount (DEFECT 1)')
assert.doesNotMatch(inferenceTelemetryHookSource, /useMemo\(\(\) => createInferenceTelemetryStore/, 'per-mount store creation loses every event emitted while the view is unmounted')
assert.doesNotMatch(inferenceTelemetryHookSource, /store\.disconnect\(\)/, 'unmount must not tear down the shared stream — navigation would wipe run state (DEFECT 2)')
assert.match(appSource, /ensureInferenceTelemetryConnected/, 'the app shell must connect the observatory stream at startup, not first view mount (DEFECT 1)')

/* ---- Flow Bench (Phase 6.1) ---- */
const flowBenchSource = read('../src/components/observatory/FlowBench.jsx')
const flowBenchEngineSource = read('../src/lib/observatory/flowBench.js')
const observatoryViewSource = read('../src/views/InferenceObservatoryView.jsx')
assert.match(observatoryViewSource, /operational telemetry — not compatibility evidence/, 'the Flow Bench view must carry the telemetry-not-evidence affordance')
assert.match(observatoryViewSource, /flowbench-rail__tiles/, 'the instrument rail tiles must be present')
assert.match(flowBenchSource, /aria-hidden="true"/, 'the sim canvases must be aria-hidden; the rail and log carry the information')
assert.match(flowBenchSource, /reducedMotion/, 'reduced motion must render a static field instead of animation')
assert.match(flowBenchSource, /visibilitychange/, 'the sim must pause on document.hidden')
assert.match(flowBenchSource, /subscribeLifecycle/, 'the sim must consume the shared lifecycle bus — no separate measurement path')
assert.doesNotMatch(flowBenchEngineSource, /--color-verified|--color-evidence/, 'copper and amber are claim colors and are forbidden in the fluid')
assert.doesNotMatch(flowBenchSource, /promptText|messageContent|\.content\b/, 'the sim consumes counts and timings only, never content')
assert.match(telemetryLogSource, /export function beginRequest/, 'request ids must be minted at send time so sim and metrics logs match one-to-one')

/* ---- Command palette + shortcuts (Phase 7) ---- */
const paletteSource = read('../src/components/CommandPalette.jsx')
const frontendReadmeSource = read('../README.md')
assert.match(appSource, /<CommandPalette/, 'the app must mount the command palette')
assert.match(appSource, /<ShortcutsOverlay/, 'the app must mount the shortcuts overlay')
assert.match(appSource, /lazy\(\(\) => import\('\.\/views\//, 'non-chat views must stay route-split')
assert.match(paletteSource, /readiness still gates send/, 'palette model switching must stay gate-honest')
assert.match(paletteSource, /camelid:open-ledger/, 'palette ledger jumps must use the shared deep-link event')
assert.match(frontendReadmeSource, /readiness-gate semantics are \*\*unchanged\*\*/, 'frontend README must state gate semantics are unchanged after the overhaul')

/* ---- Session telemetry (Phase 6) ---- */
assert.match(telemetryViewSource, /operational telemetry — not compatibility evidence/, 'every telemetry surface must carry the not-evidence affordance')
assert.match(telemetryViewSource, /useState\(false\)/, 'prompt reveal must default to redacted')
assert.match(telemetryViewSource, /•••• redacted/, 'redacted prompts must render visibly redacted')
assert.match(telemetryViewSource, /It never seeds or invents data/, 'the empty state must promise no synthetic data')
assert.doesNotMatch(telemetryViewSource, /EvidenceChip[\s\S]{0,300}(ttftMs|tokensPerSec|durationMs|medianT)/, 'perf numbers must never render inside Evidence Chips')
assert.doesNotMatch(telemetryLogSource, /Math\.random|seedData|sampleData|fakeData|demoData/, 'the telemetry store must have no synthetic data path')
assert.match(dashboardHookSource, /recordChatGeneration\(/, 'chat sends must feed the session telemetry store')
assert.match(dashboardHookSource, /recordHealthPoll\(/, 'health polls must feed the reachability history')
assert.match(apiWorkbenchSource, /recordWorkbenchRun\(/, 'workbench try-its must feed the session telemetry store')

/* Behavioral: export is path/content-free by whitelist even for salted records. */
const { recordChatGeneration: telRecord, exportTelemetryJson: telExport } = await import('../src/lib/telemetryLog.js')
telRecord({ modelId: 'salt-model', durationMs: 12, ttftMs: 5, outcome: 'ok', promptText: 'SECRET PROMPT /Volumes/Untitled/models/secret.gguf' })
const telExported = telExport()
assert.doesNotMatch(telExported, /SECRET PROMPT|\/Volumes\/|promptText/, 'telemetry exports must exclude prompt content and paths by whitelist')
assert.match(telExported, /salt-model/, 'telemetry exports keep whitelisted fields')
assert.match(telExported, /Not compatibility or support evidence/, 'telemetry exports must carry the not-evidence note')

/* ---- API workbench (Phase 5) ---- */
assert.match(apiViewSource, /<ApiWorkbench/, 'the API view must mount the workbench')
assert.match(apiViewSource, /chatUnlocked=\{selectedExactRowReady\}/, 'workbench generation gating must come from the shared exact-row chat gate')
assert.match(apiWorkbenchSource, /Requires a loaded supported model/, 'gated generation try-its must say they require a loaded supported model')
assert.match(apiWorkbenchSource, /gated exactly like chat/, 'the guarded copy must tie the workbench gate to the chat gate')
assert.match(apiWorkbenchSource, /operational telemetry — not compatibility evidence/, 'the request inspector must carry the telemetry-not-evidence banner')
assert.match(apiWorkbenchSource, /fail_closed/, 'fail-closed routes must render their typed guarded state')
assert.doesNotMatch(apiWorkbenchSource, /dangerouslySetInnerHTML/, 'inspector output must render as text')
/* lib/apiExamples.js is deliberately NOT in the brand sweep: code samples may
   name the SDK class they instantiate (technical compatibility content); UI
   copy may not. The sweep still covers the workbench component itself. */

/* ---- Compatibility ledger (Phase 4) ---- */
assert.match(compatibilityViewSource, /capabilities\?\.model_compatibility/, 'the ledger must render rows from the live contract only')
assert.match(compatibilityViewSource, /Not claimed/, 'the ledger must render the not-claimed column')
assert.match(compatibilityViewSource, /Resemblance is not evidence/, 'the ledger explainer must state that resemblance is not evidence')
assert.match(compatibilityViewSource, /Promotion path/, 'non-supported rows must show their promotion path from contract next_step copy')
assert.doesNotMatch(compatibilityViewSource, /supported_exact_row_smoke|supported_current_gate|tinyllama_|llama32_|llama3_|mistral|mixtral/i, 'the ledger source must contain zero hardcoded row ids or support statuses — the contract is the only voice')
assert.match(evidenceChipSource, /camelid:open-ledger/, 'Evidence Chips must deep-link to the ledger via the open-ledger event')
assert.match(appSource, /camelid:open-ledger/, 'the app shell must listen for ledger deep-links')
assert.match(modelsViewSource, /setTab\('compatibility'\)/, 'unmatched model cards must link to the compatibility ledger view')

/* ---- Analytics ---- */
assert.match(analyticsViewSource, /displayCapabilityId\(feature\.id\)/, 'Analytics view should not render raw provider-scoped API feature ids')

/* ---- TopBar (re-baselined to the Evidence Chip gate) ---- */
assert.match(topBarSource, /getChatGateState\(capabilities, selectedModel, runtime\)/, 'TopBar must derive its support claim from the shared chat gate')
assert.match(topBarSource, /<EvidenceChip/, 'TopBar support gate must render through the Evidence Chip')
assert.match(topBarSource, /className="topbar__gate"/, 'TopBar gate block must render on every tab, not only chat')
assert.doesNotMatch(topBarSource, /tab === 'chat' && !demoMode &&[\s\S]*topbar__gate/, 'TopBar gate visibility must not be restricted to the chat tab')
assert.match(topBarSource, /state=\{gate\.contractSupported \? 'supported'/, 'TopBar Evidence Chip supported state must come only from the shared gate contract flag')

/* ---- Evidence Chip system (Phase 1 contract) ---- */
assert.doesNotMatch(evidenceChipSource, /fetch\(|getChatGateState|isCompatibilitySupportedForModel|findCompatibilityHint/, 'EvidenceChip must stay purely presentational — it renders gate state, never computes or fetches it')
assert.match(evidenceStatusSource, /if \(value === 'supported' \|\| value\.startsWith\('supported_'\)\) return 'supported'/, 'only contract supported/supported_* statuses may classify into the copper supported state')
assert.match(evidenceCss, /\.ev-chip--supported\s*\{[^}]*var\(--color-verified\)/s, 'the supported chip state must use the reserved copper tokens')
for (const [path, css] of statusSheets) {
  assert.doesNotMatch(css, /var\(--color-verified\)/, `${path} must not spend the copper supported color on non-claim surfaces`)
}

/* ---- Tokens, fonts, themes (Phase 1 contract) ---- */
assert.doesNotMatch(tokensCss, /@import url\(['"]?https?:/, 'tokens.css must not import from a CDN — the app renders fully offline')
assert.match(tokensCss, /:root \{\s*\n\s*color-scheme: dark;/, 'dark is the canonical :root palette (dark-first)')
assert.match(tokensCss, /--color-verified:/, 'tokens must define the reserved copper supported color')
assert.match(tokensCss, /--color-evidence:/, 'tokens must define the bounded-evidence amber distinct from copper')
assert.match(mainSource, /@fontsource-variable\/inter/, 'body font must be self-hosted via Fontsource')
assert.match(mainSource, /@fontsource\/ibm-plex-mono/, 'mono font must be self-hosted via Fontsource')
assert.doesNotMatch(mainSource, /fonts\.googleapis|fonts\.gstatic/, 'no third-party font CDN calls')
assert.match(useThemeSource, /return saved && VALID\.has\(saved\) \? saved : 'dark'/, 'theme preference must default to dark')

/* ---- Brand hygiene across visible UI sources ---- */
const visibleUiSources = [
  '../src/views/ChatWorkspace.jsx',
  '../src/views/ApiView.jsx',
  '../src/views/SystemView.jsx',
  '../src/views/ModelsView.jsx',
  '../src/hooks/useDashboardData.js',
  '../src/components/TopBar.jsx',
  '../src/components/chat/MessageTurn.jsx',
  '../src/components/ui/EvidenceChip.jsx',
  '../src/lib/evidenceStatus.js',
  '../src/lib/markdown.jsx',
  '../src/components/models/ModelInspector.jsx',
  '../src/components/models/TokenizerPlayground.jsx',
  '../src/views/CompatibilityView.jsx',
  '../src/components/api/ApiWorkbench.jsx',
  '../src/views/TelemetryView.jsx',
].map((path) => [path, read(path)])
for (const [path, source] of visibleUiSources) {
  assert.doesNotMatch(source, /\b(OpenAI|ChatGPT|Claude|Gemini)\b/, `${path} visible copy should not mention competitor brands`)
}

/* ---- Streaming visuals (current chat.css/ui.css truth) ---- */
assert.match(chatCss, /\.streaming-loader\s*\{[^}]*display:\s*inline-flex/s, 'streaming assistant rows should keep a dedicated loader')
assert.match(chatCss, /\.streaming-loader-dot\s*\{[^}]*border-radius:\s*50%[^}]*animation:\s*camelidDotBounce/s, 'streaming loader dots should animate only while the loader is rendered')
assert.match(chatCss, /\.streaming-loader-compact\s*\{[^}]*padding:\s*0 0 8px/s, 'compact streaming loader should sit above pre-token assistant content without extra copy')
assert.match(chatCss, /\.message-code-card\.is-generating\s*\{/, 'incomplete streaming code cards should have an active visual treatment')
assert.match(chatCss, /\.message-live-generation-badge\s*\{/, 'streaming assistant content should keep a visible active badge while the backend is generating')
assert.match(chatCss, /\.message-live-dot\s*\{[^}]*animation:\s*cxPulse/s, 'live generation badges should visibly pulse only while the badge is rendered')
assert.match(uiCss, /@keyframes cxPulse/, 'the live pulse keyframes must exist')
assert.match(tokensCss, /@keyframes camelidDotBounce/, 'the streaming dot bounce keyframes must exist')

console.log('UI regression smoke passed (re-baselined Phase 2 pre-work)')
