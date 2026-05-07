import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, formatCapabilityStatus, getCurrentCompatibilityTarget, isCompatibilitySupportedForModel, isGuardedCapabilityStatus, isSupportedCapabilityStatus } from '../lib/capabilities'
import { clampText, formatDate, formatRate } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel, isRunnableInCurrentRuntime } from '../lib/modelState'

const CHAT_DEMO_TOKEN_CAP = 16

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const formatProbability = (value) => {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${(number * 100).toFixed(number >= 0.1 ? 1 : 2)}%`
}

const summarizeGuardedFeatures = (features) => {
  const guarded = features.filter((feature) => isGuardedCapabilityStatus(feature.status))
  if (!guarded.length) return 'No unsupported or partial API rows advertised by /api/capabilities.'
  const summary = guarded.slice(0, 3).map((feature) => `${feature.id}: ${formatCapabilityStatus(feature.status)}`).join(' · ')
  return `${guarded.length} guarded API feature${guarded.length === 1 ? '' : 's'}: ${summary}${guarded.length > 3 ? ' · …' : ''}`
}

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  models,
  runtime,
  capabilities,
  latestAssistantMessage,
  pendingConversation,
  composer,
  setComposer,
  saveToMemory,
  sendMessage,
  sending,
  selectedModelRunnable,
  setTab,
}) {
  const visibleMessages = (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message))
  const pendingPrompt = (pendingConversation?.content || (sending ? composer.trim() : '')).trim()
  const pendingPromptAlreadyVisible = Boolean(
    pendingPrompt && [...visibleMessages].reverse().some((message) => message.role === 'user' && message.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const awaitingAssistant = Boolean(sending && pendingPrompt)
  const isFreshThread = selectedConversation ? (visibleMessages.length === 0 && !pendingPrompt) : !pendingPrompt
  const latestVisibleAssistantMessage = [...visibleMessages].reverse().find((message) => message.role === 'assistant') || latestAssistantMessage

  const handleComposerKeyDown = async (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) {
        await sendMessage()
      }
    }
  }

  const rawConversationTitle = selectedConversation?.title?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const conversationLabel = clampText(hasCustomConversationTitle ? rawConversationTitle : 'Untitled chat', 30)
  const lastUpdated = selectedConversation?.updated_at ? formatDate(selectedConversation.updated_at) : null
  const latestTelemetryMatchesSelection = !latestVisibleAssistantMessage?.model_id || latestVisibleAssistantMessage.model_id === selectedModelId
  const latestTelemetryMessage = latestTelemetryMatchesSelection ? latestVisibleAssistantMessage : null
  const speedLabel = latestTelemetryMessage?.tokens_out_per_sec !== null && latestTelemetryMessage?.tokens_out_per_sec !== undefined
    ? formatRate(latestTelemetryMessage.tokens_out_per_sec)
    : 'Waiting for first reply'
  const latestTopLogits = (latestTelemetryMessage?.top_logits || []).slice(0, 5)
  const latestGeneratedTokens = latestTelemetryMessage?.usage?.completion_tokens
  const latestFirstGeneratedToken = latestTelemetryMessage?.generated_token_ids?.[0]
  const latestFirstTokenCopy = latestFirstGeneratedToken !== null && latestFirstGeneratedToken !== undefined ? ` · first token #${latestFirstGeneratedToken}` : ''
  const latestCompletionCopy = latestGeneratedTokens === 1
    ? `1 completion token${latestFirstTokenCopy} · first-token path validated; longer replies still need polish`
    : latestGeneratedTokens
      ? `${latestGeneratedTokens} completion tokens${latestFirstTokenCopy} · raw ${latestTelemetryMessage?.demo_token_cap || CHAT_DEMO_TOKEN_CAP}-token-cap local run; inspect before trusting polish`
      : `First reply will establish the live TPS baseline for this loaded model (${CHAT_DEMO_TOKEN_CAP}-token demo cap).`
  const staleTelemetryModelLabel = latestVisibleAssistantMessage?.model_id && !latestTelemetryMatchesSelection
    ? (latestVisibleAssistantMessage.model_name || latestVisibleAssistantMessage.model_id)
    : ''
  const runnableModels = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked)
  const hasRunnableChoices = runnableModels.length > 0
  const modelPickerTitle = selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what Camelid should use for this chat.'
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedRuntimeReady = selectedChatGate.runtimeReady || isRunnableInCurrentRuntime(selectedModel, runtime)
  const selectedModelCapabilitySupported = selectedChatGate.contractSupported || isCompatibilitySupportedForModel(capabilities, selectedModel)
  const supportBlocked = selectedRuntimeReady && !selectedModelCapabilitySupported
  const selectedModelMeta = supportBlocked
    ? 'Loaded, but not supported by the current compatibility contract'
    : !selectedModelRunnable
      ? describeModelState(selectedModel)
      : runtime?.loaded_now && runtime?.active_model_id === selectedModelId
      ? (isFreshThread ? 'Loaded + generation-ready' : speedLabel)
      : isFreshThread
        ? 'Ready to chat'
        : speedLabel
  const canSubmit = Boolean(composer.trim()) && selectedModelRunnable && !sending
  const supportContract = capabilities?.support_contract
  const apiFeatures = capabilities?.api_features || []
  const chatFeature = apiFeatures.find((feature) => feature.id === 'openai_chat_completions')
  const currentCompatibilityTarget = getCurrentCompatibilityTarget(capabilities)
  const supportedCompatibilityRows = (capabilities?.model_compatibility || []).filter((target) => isSupportedCapabilityStatus(target.status))
  const supportedCompatibilitySummary = supportedCompatibilityRows.map((target) => target.id).join(' · ')
  const selectedCompatibilityHint = findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = selectedCompatibilityHint?.kind === 'compatibility' ? selectedCompatibilityHint.target : null
  const selectedCompatibilitySupported = selectedCompatibilityTarget ? isSupportedCapabilityStatus(selectedCompatibilityTarget.status) : false
  const capabilityGate = supportContract?.current_gate || 'No /api/capabilities contract'
  const compatibilityLabel = supportedCompatibilitySummary || (currentCompatibilityTarget
    ? `${currentCompatibilityTarget.id} · ${formatCapabilityStatus(currentCompatibilityTarget.status)}`
    : 'No compatibility target advertised')
  const selectedCompatibilityLabel = selectedModel
    ? compatibilityHintLabel(selectedCompatibilityHint, 'No matching COMPATIBILITY.md row')
    : 'No model selected'
  const selectedCompatibilityCopy = selectedModel
    ? compatibilityHintCopy(selectedCompatibilityHint)
    : 'Choose a model before inferring any support boundary. Camelid will not promote filenames or saved paths into compatibility claims.'
  const compatibilityEvidence = supportedCompatibilitySummary
    ? `Supported rows: ${supportedCompatibilitySummary}. Runtime loaded_now=true and generation_ready=true are still required.`
    : currentCompatibilityTarget
      ? currentCompatibilityTarget.evidence || currentCompatibilityTarget.next_step
      : 'Camelid will not infer model-family or quantization support from filenames or saved browser paths.'
  const chatFeatureCopy = chatFeature
    ? `${formatCapabilityStatus(chatFeature.status)} · ${chatFeature.notes}`
    : 'Chat capability was not advertised; health and typed backend errors remain the source of truth.'
  const guardedFeatureSummary = summarizeGuardedFeatures(apiFeatures)
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const emptyHeroTitle = selectedModelRunnable
    ? 'Ask Camelid anything local.'
    : supportBlocked
      ? 'Exact row required.'
      : 'Load a proven local model.'
  const emptyHeroSummary = selectedModelRunnable
    ? 'A quiet, Gemini-like chat surface for bounded local replies. Readiness stays visible without turning filenames into support claims.'
    : supportBlocked
      ? 'The runtime sees a loaded GGUF, but chat stays locked until /api/capabilities matches its exact supported row.'
      : 'Choose a GGUF, then Camelid unlocks chat only when health and the compatibility contract agree on that exact model.'
  const readinessFacts = [
    {
      label: 'Runtime',
      value: selectedModel ? (selectedRuntimeReady ? 'Ready' : 'Waiting') : 'No model',
      copy: 'active_model_id + loaded_now + generation_ready',
    },
    {
      label: 'Contract',
      value: selectedModelRunnable ? 'Matched' : supportBlocked ? 'Missing row' : capabilityGate,
      copy: selectedModel ? selectedCompatibilityLabel : 'No inferred compatibility before selection',
    },
    {
      label: 'Boundary',
      value: 'Exact evidence only',
      copy: 'No filename, path, or neighboring-family optimism',
    },
  ]
  const proofPills = [
    selectedModelRunnable ? 'Chat unlocked' : 'Chat guarded',
    selectedModel ? selectedModelName : 'Local GGUF required',
    `Demo cap ${CHAT_DEMO_TOKEN_CAP} tokens`,
  ]

  const renderCapabilityStrip = (stage = false) => (
    <div className={`chat-capability-strip ${stage ? 'chat-capability-strip-stage' : ''}`} aria-label="Camelid support contract and chat readiness">
      <div>
        <span>Support gate</span>
        <strong>{capabilityGate}</strong>
        <small>{compatibilityLabel}</small>
      </div>
      <div>
        <span>Chat unlock</span>
        <strong>{selectedModelRunnable ? 'loaded_now=true + generation_ready=true + exact compatibility row' : supportBlocked ? 'Blocked by exact-row compatibility contract' : 'Blocked until health is ready'}</strong>
        <small>loaded_now={runtime?.loaded_now ? 'true' : 'false'} · generation_ready={runtime?.generation_ready ? 'true' : 'false'}; chat requires active_model_id to equal the selected local GGUF and match an exact supported COMPATIBILITY.md row.</small>
      </div>
      <div>
        <span>API guardrails</span>
        <strong>{chatFeatureCopy}</strong>
        <small>{guardedFeatureSummary}</small>
      </div>
      {stage && (
        <>
          <div>
            <span>Selected model contract</span>
            <strong>{selectedCompatibilityLabel}</strong>
            <small>{selectedCompatibilitySupported ? selectedCompatibilityCopy : `${selectedCompatibilityCopy} Chat still requires the runtime health gate.`}</small>
          </div>
          <div>
            <span>Evidence note</span>
            <strong>No filename optimism</strong>
            <small>{compatibilityEvidence}</small>
            <button type="button" className="ghost-button ghost-button-quiet" onClick={() => setTab('api')}>Open API contract</button>
          </div>
        </>
      )}
    </div>
  )

  const renderModelPicker = () => {
    if (!hasRunnableChoices) {
      return (
        <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>
          Choose model
        </button>
      )
    }

    return (
      <label className="composer-model-picker" title={modelPickerTitle}>
        <span className="composer-tool-label">Model</span>
        <select
          className="composer-model-select"
          aria-label="Choose model for chat"
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={sending}
        >
          {runnableModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
    )
  }

  return (
    <section className={`chat-layout chat-layout-gemini view-stack ${isFreshThread ? 'chat-layout-empty' : ''}`}>
      {selectedConversation && (
        <div className="mobile-conversation-bar" aria-label="Conversation navigation">
          <button className="ghost-button mobile-conversation-trigger" onClick={() => setTab('history')}>
            <span>Conversations</span>
            <strong title={rawConversationTitle || 'Untitled chat'}>{conversationLabel}</strong>
          </button>
          <div className="mobile-conversation-status">
            {lastUpdated ? `Updated ${lastUpdated}` : 'Current thread'}
          </div>
        </div>
      )}

      <div className={`chat-canvas ${isFreshThread ? 'chat-canvas-empty' : ''}`}>
        {isFreshThread ? (
          <div className="chat-empty-shell chat-empty-shell-gemini">
            <div className="chat-empty-stage chat-empty-stage-clean">
              <div className="chat-empty-hero chat-empty-hero-gemini chat-empty-hero-clean">
                <div className="chat-empty-orb" aria-hidden="true">
                  <span>C</span>
                  <i />
                </div>
                <p className="chat-empty-greeting">Camelid local chat</p>
                <h2>{emptyHeroTitle}</h2>
                <p className="hero-summary">{emptyHeroSummary}</p>
                <div className="chat-empty-proofbar" aria-label="Current chat guard summary">
                  {proofPills.map((pill) => <span key={pill}>{pill}</span>)}
                </div>
                <p className="chat-empty-contract-note">Chat opens only when /v1/health and /api/capabilities agree on the selected exact GGUF.</p>
              </div>

              <div className="chat-empty-readiness chat-empty-readiness-ledger" aria-label="Local chat readiness summary">
                {readinessFacts.map((item) => (
                  <div key={item.label} className="chat-empty-readiness-card">
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                    <small>{item.copy}</small>
                  </div>
                ))}
              </div>

              <div className="composer composer-gemini composer-gemini-stage composer-gemini-stage-clean">
                <textarea className="composer-input composer-input-gemini composer-input-gemini-stage" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={2} placeholder={selectedModelRunnable ? 'Message Camelid…' : 'Select a ready model first'} disabled={sending || !selectedModelRunnable} />
                <div className="composer-gemini-footer composer-gemini-footer-stage composer-gemini-footer-stage-clean">
                  <div className="composer-gemini-tools composer-gemini-tools-stage composer-gemini-tools-stage-clean">
                    {renderModelPicker()}
                    {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>Open Library</button>}
                  </div>
                  <div className="composer-gemini-actions composer-gemini-actions-stage">
                    <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{sending ? 'Sending…' : 'Send'}</button>
                  </div>
                </div>
              </div>

              <p className="chat-empty-status-note">{selectedModelMeta}</p>
            </div>
          </div>
        ) : (
          <>
            {renderCapabilityStrip()}

            {selectedModelRunnable && (
              <div className="precision-strip panel" aria-label="Camelid precision telemetry">
                <div className="precision-speed-card">
                  <span>Last local reply speed</span>
                  <strong>{speedLabel}</strong>
                  <small>{staleTelemetryModelLabel ? `Last reply used ${staleTelemetryModelLabel}. Send one prompt with ${selectedModel?.name || 'this model'} to refresh telemetry.` : latestCompletionCopy}</small>
                </div>
                <div className="precision-logit-card">
                  <div className="precision-logit-head">
                    <span>Diagnostic logits</span>
                    <strong>Top-5 first-token probabilities</strong>
                  </div>
                  {latestTopLogits.length > 0 ? (
                    <div className="logit-list" role="list">
                      {latestTopLogits.map((entry) => (
                        <div key={`${entry.rank}-${entry.token_id}`} className={`logit-row ${entry.selected ? 'selected' : ''}`} role="listitem">
                          <span className="logit-rank">#{entry.rank}</span>
                          <code title={`token ${entry.token_id}`}>{entry.text || `#${entry.token_id}`}</code>
                          <span className="logit-probability">{formatProbability(entry.probability)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="precision-empty">{staleTelemetryModelLabel ? 'Telemetry is hidden here because it belongs to a different selected model. Send a fresh prompt to avoid stale readouts.' : 'Awaiting the next completion. The backend returns first-token top logits with completed responses; streaming token-level probabilities are not wired yet.'}</p>
                  )}
                </div>
              </div>
            )}

            {!selectedModelRunnable && (
              <div className="setup-card setup-card-inline setup-card-gemini">
                <div>
                  <p className="panel-kicker">Before you chat</p>
                  <h2>{supportBlocked ? 'Support contract needs an exact row' : 'Choose a runnable model'}</h2>
                  <p className="hero-summary">{supportBlocked ? `${selectedCompatibilityLabel}. ${selectedCompatibilityCopy}` : describeModelState(selectedModel)}</p>
                </div>
                <div className="composer-actions single-action-row">
                  <button className="primary-button" onClick={() => setTab('library')}>Open Library</button>
                </div>
              </div>
            )}

            <div className="chat-thread chat-thread-gemini">
              {visibleMessages.length === 0 && !awaitingAssistant && <div className="empty-state empty-state-chat">Pick a ready model, then send the first message when you’re ready.</div>}
              {visibleMessages.map((message) => {
                const hasMetrics = message.role === 'assistant' && (message.tokens_in_per_sec !== null && message.tokens_in_per_sec !== undefined || message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined)
                const messageCompletionTokens = message.usage?.completion_tokens
                const modelLabel = message.model_name || message.model_id
                const demoTokenCap = message.demo_token_cap || CHAT_DEMO_TOKEN_CAP
                const firstGeneratedToken = message.generated_token_ids?.[0]
                const firstTokenCopy = firstGeneratedToken !== null && firstGeneratedToken !== undefined ? ` First token #${firstGeneratedToken}.` : ''
                const diagnosticCopy = message.role === 'assistant'
                  ? messageCompletionTokens === 1
                    ? `Raw first-token validation sample.${firstTokenCopy} Longer generation is not polished yet.`
                    : messageCompletionTokens
                      ? `Raw local output · ${messageCompletionTokens} completion tokens${messageCompletionTokens >= demoTokenCap ? ' (demo cap)' : ''}.${firstTokenCopy} Longer-generation polish still needs separate validation.`
                      : 'Raw local output.'
                  : ''

                return (
                  <article key={message.id} className={`message-row message-row-gemini ${message.role}`}>
                    <div className={`message-bubble message-bubble-gemini ${message.role}`}>
                      {message.role === 'assistant' && (
                        <div className="message-heading message-heading-clean">
                          <span className="message-micro-meta">{[modelLabel, hasMetrics ? `${formatRate(message.tokens_out_per_sec)} out` : 'raw local reply'].filter(Boolean).join(' · ')}</span>
                        </div>
                      )}
                      <p>{message.content}</p>
                      {message.role === 'assistant' && (diagnosticCopy || hasMetrics) && (
                        <div className="message-footnote">
                          {diagnosticCopy && <span>{diagnosticCopy}</span>}
                          {message.tokens_in_per_sec !== null && message.tokens_in_per_sec !== undefined && <span>In {formatRate(message.tokens_in_per_sec)}</span>}
                          {message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined && <span>Out {formatRate(message.tokens_out_per_sec)}</span>}
                        </div>
                      )}
                      {message.role === 'assistant' && message.top_logits?.length > 0 && (
                        <div className="message-logit-viewer" aria-label="Top five first-token probabilities for this reply">
                          <div className="message-logit-title">Top-5 first-token probabilities</div>
                          {message.top_logits.slice(0, 5).map((entry) => (
                            <div key={`${message.id}-${entry.rank}-${entry.token_id}`} className="message-logit-row">
                              <span>#{entry.rank}</span>
                              <code title={`token ${entry.token_id}`}>{entry.text || `#${entry.token_id}`}</code>
                              <strong>{formatProbability(entry.probability)}</strong>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </article>
                )
              })}
              {awaitingAssistant && (
                <>
                  {pendingUserPrompt && (
                    <article className="message-row message-row-gemini user pending">
                      <div className="message-bubble message-bubble-gemini user pending">
                        <p>{pendingUserPrompt}</p>
                      </div>
                    </article>
                  )}
                  <article className="message-row message-row-gemini assistant pending">
                    <div className="message-thinking-loader camelid-walk-loader" aria-hidden="true">
                      <span className="camelid-walk-ground" />
                      <span className="camelid-walk-body" />
                      <span className="camelid-walk-hump" />
                      <span className="camelid-walk-neck" />
                      <span className="camelid-walk-head" />
                      <span className="camelid-walk-ear" />
                      <span className="camelid-walk-tail" />
                      <span className="camelid-walk-leg camelid-walk-leg-1" />
                      <span className="camelid-walk-leg camelid-walk-leg-2" />
                      <span className="camelid-walk-leg camelid-walk-leg-3" />
                      <span className="camelid-walk-leg camelid-walk-leg-4" />
                    </div>
                    <div className="message-bubble message-bubble-gemini assistant pending">
                      <div className="message-heading message-heading-clean">
                        <span className="message-micro-meta">Thinking…</span>
                      </div>
                      <p className="message-placeholder-copy">Generating a raw local reply with a {CHAT_DEMO_TOKEN_CAP}-token demo cap… first-token diagnostics appear after completion.</p>
                    </div>
                  </article>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {!isFreshThread && (
        <div className="composer composer-gemini composer-gemini-floating">
          <textarea className="composer-input composer-input-gemini" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={3} placeholder={selectedModelRunnable ? 'Ask a short local test prompt' : 'Pick a ready model first, then start your chat'} disabled={sending || !selectedModelRunnable} />
          <div className="composer-gemini-footer">
            <div className="composer-gemini-tools">
              {renderModelPicker()}
              <span className="composer-meta-pill">{selectedModelMeta}</span>
              {selectedModelRunnable && <button className="ghost-button subtle-action" onClick={saveToMemory} disabled={sending}>Save to memory</button>}
            </div>
            <div className="composer-gemini-actions">
              {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button" onClick={() => setTab('library')}>Open Library</button>}
              <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{sending ? 'Sending…' : 'Send'}</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
