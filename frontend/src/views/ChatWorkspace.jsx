import { useEffect, useState } from 'react'

import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel } from '../lib/capabilities'
import { clampText, formatDate, formatRate } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel, isRunnableInCurrentRuntime } from '../lib/modelState'

const isBootstrapMessage = (message) =>
  message?.role === 'assistant' &&
  typeof message?.content === 'string' &&
  message.content.startsWith('Conversation created.')

const cleanLegacyDemoCapCopy = (value) => {
  if (typeof value !== 'string') return value
  return value
    .replace(/\s*\(demo cap\)/gi, '')
    .replace(/\s*·\s*raw\s+16-token-cap\s+local\s+run;\s*inspect\s+before\s+trusting\s+polish/gi, ' · raw local run')
    .replace(/\s*Longer-generation\s+polish\s+still\s+needs\s+separate\s+validation\.?/gi, '')
    .replace(/\s*Longer\s+generation\s+is\s+not\s+polished\s+yet\.?/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

const normalizeCodeLanguage = (value) => {
  const language = String(value || '').trim().replace(/[^a-zA-Z0-9_+#.-].*$/, '')
  if (!language) return 'Code'
  if (language.toLowerCase() === 'js') return 'JavaScript'
  if (language.toLowerCase() === 'ts') return 'TypeScript'
  if (language.toLowerCase() === 'html') return 'HTML'
  if (language.toLowerCase() === 'css') return 'CSS'
  return language.toUpperCase()
}

const copyText = async (text) => {
  try {
    await navigator.clipboard?.writeText(text)
  } catch {
    // Clipboard access can be denied outside secure browser contexts; rendering still works.
  }
}

const renderInlineMarkdown = (text, keyPrefix) => {
  const parts = String(text || '').split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean)
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={key} className="inline-code">{part.slice(1, -1)}</code>
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }
    return <span key={key}>{part}</span>
  })
}

const normalizeProseForReading = (text) => String(text || '')
  .replace(/\r\n/g, '\n')
  .replace(/\s+(Page\s+\d+\b)/gi, '\n\n$1')
  .replace(/\s+(References?\s*:)/gi, '\n\n$1')
  .replace(/\s+(Works\s+Cited\s*:)/gi, '\n\n$1')
  .replace(/\s+([•*-]\s+["“])/g, '\n$1')

const splitLongParagraph = (value) => {
  const text = String(value || '').trim()
  if (text.length <= 520) return text ? [text] : []
  const sentences = text.match(/[^.!?]+[.!?]+["”']?|[^.!?]+$/g) || [text]
  const paragraphs = []
  let current = ''

  sentences.forEach((sentence) => {
    const next = `${current}${current ? ' ' : ''}${sentence.trim()}`.trim()
    if (current && (next.length > 620 || current.split(/[.!?]+/).filter(Boolean).length >= 4)) {
      paragraphs.push(current)
      current = sentence.trim()
    } else {
      current = next
    }
  })
  if (current) paragraphs.push(current)
  return paragraphs
}

const pushParagraphBlocks = (blocks, value, keyPrefix) => {
  splitLongParagraph(value).forEach((paragraph) => {
    blocks.push(<p key={`${keyPrefix}-p-${blocks.length}`}>{renderInlineMarkdown(paragraph, `${keyPrefix}-p-${blocks.length}`)}</p>)
  })
}

const renderMarkdownText = (text, keyPrefix) => {
  const lines = normalizeProseForReading(text).split('\n')
  const blocks = []
  let paragraph = []
  let list = []

  const flushParagraph = () => {
    if (paragraph.length) {
      const value = paragraph.join(' ').trim()
      if (value) {
        pushParagraphBlocks(blocks, value, keyPrefix)
      }
      paragraph = []
    }
  }
  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={`${keyPrefix}-ul-${blocks.length}`}>
          {list.map((item, index) => (
            <li key={`${keyPrefix}-li-${blocks.length}-${index}`}>{renderInlineMarkdown(item, `${keyPrefix}-li-${index}`)}</li>
          ))}
        </ul>,
      )
      list = []
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushList()
      return
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      flushParagraph()
      flushList()
      const Tag = heading[1].length === 1 ? 'h2' : 'h3'
      blocks.push(<Tag key={`${keyPrefix}-h-${blocks.length}`}>{renderInlineMarkdown(heading[2], `${keyPrefix}-h-${blocks.length}`)}</Tag>)
      return
    }
    const pageHeading = line.match(/^(Page\s+\d+)\b[:\s.-]*(.*)$/i)
    if (pageHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-page-${blocks.length}`}>{pageHeading[1]}</h3>)
      if (pageHeading[2]) {
        pushParagraphBlocks(blocks, pageHeading[2], keyPrefix)
      }
      return
    }
    const referencesHeading = line.match(/^(References?|Works\s+Cited)\s*:?(.*)$/i)
    if (referencesHeading) {
      flushParagraph()
      flushList()
      blocks.push(<h3 className="message-section-heading" key={`${keyPrefix}-ref-${blocks.length}`}>{referencesHeading[1]}</h3>)
      if (referencesHeading[2]) {
        pushParagraphBlocks(blocks, referencesHeading[2].replace(/^\s*[:*-]\s*/, ''), keyPrefix)
      }
      return
    }
    const listItem = line.match(/^[-*]\s+(.+)$/)
    if (listItem) {
      flushParagraph()
      list.push(listItem[1])
      return
    }
    flushList()
    paragraph.push(line)
  })
  flushParagraph()
  flushList()
  return blocks
}

const syntaxClassForToken = (token, language) => {
  const lowerLanguage = String(language || '').toLowerCase()
  if (/^\s+$/.test(token)) return ''
  if (/^\/\//.test(token) || /^\/\*/.test(token) || /^<!--/.test(token)) return 'comment'
  if (/^['"`]/.test(token)) return 'string'
  if (/^\d/.test(token)) return 'number'
  if (lowerLanguage.includes('html') && /^<\/?[\w-]+/.test(token)) return 'tag'
  if (lowerLanguage.includes('html') && /^[\w:-]+(?==)/.test(token)) return 'attr'
  if (/^(const|let|var|function|return|if|else|for|while|class|new|true|false|null|undefined|import|export|from|async|await|document|window)$/.test(token)) return 'keyword'
  if (lowerLanguage.includes('css') && /^[\w-]+(?=\s*:)/.test(token)) return 'attr'
  return ''
}

const renderHighlightedCode = (code, language, keyPrefix) => {
  const lowerLanguage = String(language || '').toLowerCase()
  const pattern = lowerLanguage.includes('html')
    ? /(<!--[\s\S]*?-->|<\/?[\w-]+|\/?>|[\w:-]+(?==)|"(?:\\.|[^"])*"|'(?:\\.|[^'])*')/g
    : lowerLanguage.includes('css')
      ? /(\/\*[\s\S]*?\*\/|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|#[\da-fA-F]{3,8}|\b\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw)?\b|[\w-]+(?=\s*:))/g
      : /(\/\/.*|\/\*[\s\S]*?\*\/|"(?:\\.|[^"])*"|'(?:\\.|[^'])*'|`(?:\\.|[^`])*`|\b(?:const|let|var|function|return|if|else|for|while|class|new|true|false|null|undefined|import|export|from|async|await|document|window)\b|\b\d+(?:\.\d+)?\b)/g
  const nodes = []
  let cursor = 0
  let match = pattern.exec(code)
  while (match) {
    if (match.index > cursor) nodes.push(code.slice(cursor, match.index))
    const token = match[0]
    const tokenClass = syntaxClassForToken(token, lowerLanguage)
    nodes.push(tokenClass
      ? <span key={`${keyPrefix}-${nodes.length}`} className={`syntax-token ${tokenClass}`}>{token}</span>
      : token)
    cursor = match.index + token.length
    match = pattern.exec(code)
  }
  if (cursor < code.length) nodes.push(code.slice(cursor))
  return nodes
}

function AssistantMarkdown({ content }) {
  const normalized = String(content || '').replace(/\r\n/g, '\n')
  const blocks = []
  const fencePattern = /```\s*([^\n`]*)\n?([\s\S]*?)```/g
  let cursor = 0
  let match = fencePattern.exec(normalized)

  while (match) {
    const before = normalized.slice(cursor, match.index)
    blocks.push(...renderMarkdownText(before, `md-${blocks.length}`))
    const language = normalizeCodeLanguage(match[1])
    const code = match[2].replace(/^\n+|\n+$/g, '')
    blocks.push(
      <figure className="message-code-card" key={`code-${blocks.length}`}>
        <figcaption>
          <span>{language}</span>
          <button type="button" onClick={() => copyText(code)} aria-label={`Copy ${language} code`}>Copy</button>
        </figcaption>
        <pre><code>{renderHighlightedCode(code, language, `code-${blocks.length}`)}</code></pre>
      </figure>,
    )
    cursor = match.index + match[0].length
    match = fencePattern.exec(normalized)
  }
  blocks.push(...renderMarkdownText(normalized.slice(cursor), `md-${blocks.length}`))

  return <div className="message-markdown">{blocks.length ? blocks : <p>{content}</p>}</div>
}

export default function ChatWorkspace({
  selectedConversation,
  selectedModel,
  selectedModelId,
  setSelectedModelId,
  models,
  runtime,
  capabilities,
  pendingConversation,
  composer,
  setComposer,
  saveToMemory,
  sendMessage,
  sending,
  selectedModelRunnable,
  setTab,
}) {
  const [generationElapsedSeconds, setGenerationElapsedSeconds] = useState(0)
  const visibleMessages = (selectedConversation?.messages || []).filter((message) => !isBootstrapMessage(message))
  const pendingPrompt = (pendingConversation?.content || (sending ? composer.trim() : '')).trim()
  const pendingPromptAlreadyVisible = Boolean(
    pendingPrompt && [...visibleMessages].reverse().some((message) => message.role === 'user' && message.content === pendingPrompt),
  )
  const pendingUserPrompt = pendingPromptAlreadyVisible ? '' : pendingPrompt
  const awaitingAssistant = Boolean(sending && pendingPrompt)

  useEffect(() => {
    if (!sending) {
      setGenerationElapsedSeconds(0)
      return undefined
    }
    setGenerationElapsedSeconds(0)
    const startedAt = Date.now()
    const interval = window.setInterval(() => {
      setGenerationElapsedSeconds(Math.max(1, Math.floor((Date.now() - startedAt) / 1000)))
    }, 1000)
    return () => window.clearInterval(interval)
  }, [sending])

  const isFreshThread = selectedConversation ? (visibleMessages.length === 0 && !pendingPrompt) : !pendingPrompt
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
  const runnableModels = models.filter((model) => getChatGateState(capabilities, model, runtime).chatUnlocked)
  const hasRunnableChoices = runnableModels.length > 0
  const modelPickerTitle = selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what Camelid should use for this chat.'
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedRuntimeReady = selectedChatGate.runtimeReady || isRunnableInCurrentRuntime(selectedModel, runtime)
  const selectedModelCapabilitySupported = selectedChatGate.contractSupported || isCompatibilitySupportedForModel(capabilities, selectedModel)
  const supportBlocked = selectedRuntimeReady && !selectedModelCapabilitySupported
  const selectedModelMeta = supportBlocked
    ? 'Load a supported model to chat'
    : !selectedModelRunnable
      ? describeModelState(selectedModel)
      : runtime?.loaded_now && runtime?.active_model_id === selectedModelId
      ? 'Ready'
      : 'Ready to chat'
  const canSubmit = Boolean(composer.trim()) && selectedModelRunnable && !sending
  const selectedCompatibilityHint = findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityLabel = selectedModel
    ? compatibilityHintLabel(selectedCompatibilityHint, 'No matching COMPATIBILITY.md row')
    : 'No model selected'
  const selectedCompatibilityCopy = selectedModel
    ? compatibilityHintCopy(selectedCompatibilityHint)
    : 'Choose a model before inferring any support boundary. Camelid will not promote filenames or saved paths into compatibility claims.'
  const selectedModelName = selectedModel?.name || selectedModelId || 'No model selected'
  const emptyHeroEyebrow = 'Camelid'
  const readinessState = selectedModelRunnable ? 'ready' : supportBlocked ? 'blocked' : selectedModel ? 'waiting' : 'idle'
  const readinessLabel = selectedModelRunnable
    ? 'Ready'
    : supportBlocked
      ? 'Choose a supported model'
      : selectedModel
        ? 'Loading model'
        : 'Choose a model to begin'
  const productHeroTitle = selectedModelRunnable
    ? 'How can I help?'
    : supportBlocked
      ? 'Choose a supported model.'
      : 'Load a model to begin.'
  const productHeroSummary = selectedModelRunnable
    ? ''
    : supportBlocked
      ? ''
      : ''
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
            <div className={`chat-empty-stage chat-empty-stage-clean chat-empty-stage-product is-${readinessState}`}>
              <div className="chat-empty-hero chat-empty-hero-gemini chat-empty-hero-clean">
                <p className="chat-empty-greeting">{emptyHeroEyebrow}</p>
                <h2>{productHeroTitle}</h2>
                {productHeroSummary && <p className="hero-summary">{productHeroSummary}</p>}
              </div>

              <div className="composer composer-gemini composer-gemini-stage composer-gemini-stage-clean composer-gemini-product">
                <textarea className="composer-input composer-input-gemini composer-input-gemini-stage" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={2} placeholder={selectedModelRunnable ? 'Message Camelid…' : 'Load a model first'} disabled={sending || !selectedModelRunnable} />
                <div className="composer-gemini-footer composer-gemini-footer-stage composer-gemini-footer-stage-clean">
                  <div className="composer-gemini-tools composer-gemini-tools-stage composer-gemini-tools-stage-clean">
                    {renderModelPicker()}
                    {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button ghost-button-quiet" onClick={() => setTab('library')}>Open Library</button>}
                  </div>
                  <div className="composer-gemini-actions composer-gemini-actions-stage">
                    <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{sending ? `Generating ${generationElapsedSeconds}s…` : 'Send'}</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className={`chat-session-strip is-${readinessState}`} aria-label="Current Camelid chat status">
              <span className="chat-session-dot" aria-hidden="true" />
              <strong>{selectedModelName}</strong>
              <small>{selectedModelRunnable ? 'Ready when you are' : readinessLabel}</small>
            </div>

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
                const messageContent = cleanLegacyDemoCapCopy(message.content)
                const hasTokenMetrics = message.role === 'assistant' && (
                  message.tokens_in_per_sec !== null && message.tokens_in_per_sec !== undefined ||
                  message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined
                )

                return (
                  <article key={message.id} className={`message-row message-row-gemini ${message.role}`}>
                    <div className={`message-bubble message-bubble-gemini ${message.role}`}>
                      {message.role === 'assistant' ? <AssistantMarkdown content={messageContent} /> : <p>{messageContent}</p>}
                      {hasTokenMetrics && (
                        <div className="message-token-metrics" aria-label="Generation speed">
                          <span>In {formatRate(message.tokens_in_per_sec)}</span>
                          <span>Out {formatRate(message.tokens_out_per_sec)}</span>
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
                        <span className="message-micro-meta">Generating locally · {generationElapsedSeconds}s elapsed</span>
                      </div>
                      <p className="message-placeholder-copy">Camelid is running locally. Tokens will appear as they are generated.</p>
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
          <textarea className="composer-input composer-input-gemini" value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={handleComposerKeyDown} rows={3} placeholder={selectedModelRunnable ? 'Ask Camelid' : 'Load a model first'} disabled={sending || !selectedModelRunnable} />
          <div className="composer-gemini-footer">
            <div className="composer-gemini-tools">
              {renderModelPicker()}
              <span className="composer-meta-pill">{selectedModelMeta}</span>
              {selectedModelRunnable && <button className="ghost-button subtle-action" onClick={saveToMemory} disabled={sending}>Save to memory</button>}
            </div>
            <div className="composer-gemini-actions">
              {!selectedModelRunnable && hasRunnableChoices && <button className="ghost-button" onClick={() => setTab('library')}>Open Library</button>}
              <button className="primary-button composer-send-button" onClick={sendMessage} disabled={!canSubmit}>{sending ? `Generating ${generationElapsedSeconds}s…` : 'Send'}</button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
