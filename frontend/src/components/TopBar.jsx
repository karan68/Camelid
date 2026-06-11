import { memo } from 'react'
import { clampText } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { modelRuntimeIdMatches } from '../lib/modelState'
import { IconMenu } from './ui/icons'
import { StatusDot } from './ui/StatusDot'

const TITLES = {
  chat: 'Chat',
  library: 'Models',
  api: 'API',
  analytics: 'Analytics',
  history: 'Chat history',
  memory: 'Memory',
  system: 'System',
  settings: 'Settings',
  cluster: 'Cluster Topology',
  observatory: 'Inference Observatory',
}

/* Slim top bar. Chat tab shows the conversation title + a compact model status
   chip; other tabs show the view title. The full model picker and support detail
   live in the chat composer's ModelStatusChip and the System/API views. */
function TopBar({
  tab,
  setTab,
  selectedConversationTitle,
  runtime,
  capabilities,
  selectedModelId,
  models = [],
  onToggleSidebar = null,
  demoMode = false,
}) {
  const rawTitle = selectedConversationTitle?.trim()
  const hasCustomTitle = Boolean(rawTitle && rawTitle.toLowerCase() !== 'new conversation')
  const heading = tab === 'chat'
    ? (hasCustomTitle ? clampText(rawTitle, 64) : 'New chat')
    : (TITLES[tab] || 'Camelid')

  const selectedModel = models.find((m) => m.id === selectedModelId)
    || models.find((m) => modelRuntimeIdMatches(m, runtime))
  const gate = getChatGateState(capabilities, selectedModel, runtime)
  const apiUnavailable = runtime?.status === 'offline'
  const tone = gate.chatUnlocked ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warn' : 'neutral'
  const modelName = selectedModel?.name || 'No model selected'

  return (
    <header className={`topbar ${demoMode ? 'topbar--demo' : ''}`}>
      {onToggleSidebar && (
        <button type="button" className="topbar__menu" aria-label="Toggle sidebar" onClick={onToggleSidebar}>
          <IconMenu size={22} />
        </button>
      )}
      <h1 className="topbar__title" title={tab === 'chat' && hasCustomTitle ? rawTitle : heading}>{heading}</h1>
      <div className="topbar__spacer" />
      {tab === 'chat' && !demoMode && (
        <button
          type="button"
          className="topbar__model"
          onClick={() => setTab('library')}
          title={gate.chatUnlocked ? `${modelName} is ready` : 'Open Models to load or switch models'}
        >
          <StatusDot tone={tone} pulse={gate.chatUnlocked} />
          <span className="topbar__model-name">{clampText(modelName, 32)}</span>
        </button>
      )}
    </header>
  )
}

export default memo(TopBar)
