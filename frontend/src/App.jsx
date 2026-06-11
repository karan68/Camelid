import { useEffect, useMemo, useState } from 'react'
import SidebarRail from './components/layout/SidebarRail'
import TopBar from './components/TopBar'
import { BackendBanner } from './components/layout/BackendBanner'
import { Notice } from './components/ui/Notice'
import { ConfirmDialog } from './components/ui/ConfirmDialog'
import { formatPreview, formatSidebarDate } from './lib/formatters'
import { useDashboardData } from './hooks/useDashboardData'
import { useBackendLauncher } from './hooks/useBackendLauncher'
import { useNotice } from './hooks/useNotice'
import { useTheme } from './hooks/useTheme'
import ChatWorkspace from './views/ChatWorkspace'
import AnalyticsView from './views/AnalyticsView'
import HistoryView from './views/HistoryView'
import MemoryView from './views/MemoryView'
import ModelsView from './views/ModelsView'
import ApiView from './views/ApiView'
import SystemView from './views/SystemView'
import SettingsView from './views/SettingsView'
import ClusterView from './views/ClusterView'
import InferenceObservatoryView from './views/InferenceObservatoryView'

const DEMO_UI = import.meta.env?.VITE_CAMELID_DEMO_UI === 'true'
const HASH_TABS = new Set(['chat', 'library', 'api', 'analytics', 'history', 'memory', 'system', 'settings', 'cluster', 'observatory'])

function App() {
  const { notice, noticeTone, showNotice, clearNotice } = useNotice()
  const { preference, resolved, cyclePreference, setPreference } = useTheme()

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (DEMO_UI) return true
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem('camelid.sidebarCollapsed') === 'true'
  })
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(max-width: 860px)').matches,
  )
  const [pendingDeleteConversationId, setPendingDeleteConversationId] = useState(null)
  const [deleteBusy, setDeleteBusy] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined
    const media = window.matchMedia('(max-width: 860px)')
    const sync = () => { setIsMobile(media.matches); if (!media.matches) setMobileNavOpen(false) }
    media.addEventListener('change', sync)
    return () => media.removeEventListener('change', sync)
  }, [])

  const dash = useDashboardData({ showNotice, clearNotice })
  const {
    dashboard, tab, setTab, selectedConversationId, setSelectedConversationId,
    selectedModelId, setSelectedModelId, search, setSearch, memorySearch, setMemorySearch,
    composer, setComposer, newChatTitle, setNewChatTitle, sending, receiptMode, setReceiptMode,
    loadingModelId, registerForm, setRegisterForm, externalForm, setExternalForm,
    conversations, memories, filteredConversations, models, runtime, selectedConversation,
    selectedModel, selectedModelRunnable, latestAssistantMessage, pendingConversation,
    createConversation, showNewChatLanding, sendMessage, stopGeneration, saveToMemory,
    createMemory, updateMemory, deleteMemory, renameConversation, deleteConversation,
    installModel, installCatalogModel, cancelModelDownload, activateModel, unloadCurrentModel,
    registerModel, connectExternalModel, loadDashboard, stoppingGeneration,
    apiBase, setApiBase,
  } = dash

  const backend = useBackendLauncher({ showNotice, loadDashboard })

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('camelid.sidebarCollapsed', String(sidebarCollapsed))
  }, [sidebarCollapsed])

  // Deep-link the active tab via the URL hash (e.g. #library) — shareable and
  // handy for direct navigation. Runs once on mount if a valid tab hash is present.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const hash = window.location.hash.replace('#', '')
    if (HASH_TABS.has(hash)) setTab(hash)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const closeMobileNav = () => setMobileNavOpen(false)

  const navigateTab = (next) => {
    setTab(next)
    if (typeof window !== 'undefined' && HASH_TABS.has(next)) {
      window.history.replaceState(null, '', next === 'chat' ? window.location.pathname : `#${next}`)
    }
    closeMobileNav()
  }

  const selectConversation = (id) => {
    setSelectedConversationId(id)
    setTab('chat')
    closeMobileNav()
  }

  const startNewChat = () => {
    showNewChatLanding()
    closeMobileNav()
  }

  const requestDeleteConversation = (id) => {
    setPendingDeleteConversationId(id)
    setDeleteBusy(false)
  }

  const pendingDeleteConversation = useMemo(
    () => conversations.find((c) => c.id === pendingDeleteConversationId) || null,
    [conversations, pendingDeleteConversationId],
  )

  const pendingDeleteTitle = useMemo(() => {
    if (!pendingDeleteConversation) return 'Delete chat?'
    const trimmed = pendingDeleteConversation.title?.trim()
    if (trimmed && trimmed.toLowerCase() !== 'new conversation') return `Delete “${trimmed}”?`
    return `Delete untitled chat · ${formatSidebarDate(pendingDeleteConversation.updated_at) || 'new chat'}?`
  }, [pendingDeleteConversation])

  const pendingDeleteDetail = useMemo(() => {
    if (!pendingDeleteConversation) return ''
    const latest = [...(pendingDeleteConversation.messages || [])].reverse()
      .find((m) => typeof m?.content === 'string' && m.content.trim())
    const preview = formatPreview(latest?.content, 80)
    return preview === 'No messages yet' ? 'This conversation will be permanently removed.' : `“${preview}” — this conversation will be permanently removed.`
  }, [pendingDeleteConversation])

  const handleDeleteConfirm = async () => {
    if (!pendingDeleteConversationId || deleteBusy) return
    setDeleteBusy(true)
    const ok = await deleteConversation(pendingDeleteConversationId)
    if (ok) setPendingDeleteConversationId(null)
    setDeleteBusy(false)
  }

  if (!dashboard) {
    return (
      <div className="loading-shell">
        <div className="loading-shell-stack">
          <Notice notice={notice} tone={noticeTone} />
          <div>Loading Camelid…</div>
        </div>
      </div>
    )
  }

  const shellClasses = [
    'camelid-app',
    sidebarCollapsed ? 'is-collapsed' : '',
    mobileNavOpen ? 'is-mobile-open' : '',
    DEMO_UI ? 'is-demo' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className={shellClasses}>
      {!DEMO_UI && (
        <SidebarRail
          collapsed={!isMobile && sidebarCollapsed}
          onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
          showNewChatLanding={startNewChat}
          search={search}
          setSearch={setSearch}
          tab={tab}
          setTab={navigateTab}
          filteredConversations={filteredConversations}
          selectedConversationId={selectedConversation?.id || null}
          onSelectConversation={selectConversation}
          renameConversation={renameConversation}
          requestDeleteConversation={requestDeleteConversation}
          runtime={runtime}
          themePreference={preference}
          themeResolved={resolved}
          onCycleTheme={cyclePreference}
        />
      )}

      {!DEMO_UI && mobileNavOpen && (
        <button type="button" className="camelid-app__scrim" aria-label="Close navigation" onClick={closeMobileNav} />
      )}

      <main className="camelid-main" data-view={tab}>
        <TopBar
          tab={tab}
          setTab={navigateTab}
          selectedConversationTitle={selectedConversation?.title || ''}
          runtime={runtime}
          capabilities={dashboard?.capabilities}
          selectedModelId={selectedModelId}
          models={models}
          onToggleSidebar={DEMO_UI ? null : () => setMobileNavOpen((v) => !v)}
          demoMode={DEMO_UI}
        />

        {notice && (
          <div className="camelid-notice-slot">
            <Notice notice={notice} tone={noticeTone} onDismiss={clearNotice} />
          </div>
        )}

        {!DEMO_UI && runtime?.status === 'offline' && tab !== 'settings' && (
          <BackendBanner backend={backend} onOpenSettings={() => navigateTab('settings')} />
        )}

        <div className={`camelid-view ${(tab === 'chat' || tab === 'cluster' || tab === 'observatory') ? 'camelid-view--chat' : 'camelid-view--page'}`}>
          {tab === 'chat' && (
            <ChatWorkspace
              selectedConversation={selectedConversation}
              selectedModel={selectedModel}
              selectedModelId={selectedModelId}
              setSelectedModelId={setSelectedModelId}
              models={models}
              runtime={runtime}
              capabilities={dashboard?.capabilities}
              latestAssistantMessage={latestAssistantMessage}
              pendingConversation={pendingConversation}
              composer={composer}
              setComposer={setComposer}
              saveToMemory={saveToMemory}
              sendMessage={sendMessage}
              stopGeneration={stopGeneration}
              sending={sending}
              receiptMode={receiptMode}
              setReceiptMode={setReceiptMode}
              stoppingGeneration={stoppingGeneration}
              selectedModelRunnable={selectedModelRunnable}
              setTab={navigateTab}
              showNewChatLanding={startNewChat}
              demoMode={DEMO_UI}
            />
          )}

          {tab === 'analytics' && (
            <AnalyticsView conversations={conversations} models={models} runtime={runtime} capabilities={dashboard?.capabilities} />
          )}

          {tab === 'history' && (
            <HistoryView
              filteredConversations={filteredConversations}
              setSelectedConversationId={selectConversation}
              setTab={navigateTab}
              deleteConversation={requestDeleteConversation}
            />
          )}

          {tab === 'memory' && (
            <MemoryView
              memories={memories}
              memorySearch={memorySearch}
              setMemorySearch={setMemorySearch}
              selectedConversation={selectedConversation}
              latestAssistantMessage={latestAssistantMessage}
              saveToMemory={saveToMemory}
              createMemory={createMemory}
              updateMemory={updateMemory}
              deleteMemory={deleteMemory}
              setTab={navigateTab}
            />
          )}

          {tab === 'library' && (
            <ModelsView
              runtime={runtime}
              capabilities={dashboard?.capabilities}
              refreshDashboard={loadDashboard}
              registerForm={registerForm}
              setRegisterForm={setRegisterForm}
              externalForm={externalForm}
              setExternalForm={setExternalForm}
              registerModel={registerModel}
              connectExternalModel={connectExternalModel}
              models={models}
              selectedModelId={selectedModelId}
              setSelectedModelId={setSelectedModelId}
              loadingModelId={loadingModelId}
              activateModel={activateModel}
              unloadCurrentModel={unloadCurrentModel}
              installModel={installModel}
              installCatalogModel={installCatalogModel}
              cancelModelDownload={cancelModelDownload}
            />
          )}

          {tab === 'api' && <ApiView runtime={runtime} selectedModel={selectedModel} capabilities={dashboard?.capabilities} />}

          {tab === 'system' && <SystemView runtime={runtime} selectedModel={selectedModel} capabilities={dashboard?.capabilities} />}

          {tab === 'settings' && (
            <SettingsView
              runtime={runtime}
              apiBase={apiBase}
              setApiBase={setApiBase}
              backend={backend}
              showNotice={showNotice}
              themePreference={preference}
              setThemePreference={setPreference}
              onOpenCluster={() => navigateTab('cluster')}
            />
          )}

          {tab === 'cluster' && <ClusterView showNotice={showNotice} />}

          {tab === 'observatory' && <InferenceObservatoryView apiBase={apiBase} />}
        </div>
      </main>

      <ConfirmDialog
        open={Boolean(pendingDeleteConversation)}
        title={pendingDeleteTitle}
        detail={pendingDeleteDetail}
        confirmLabel="Delete"
        busy={deleteBusy}
        onCancel={() => { if (!deleteBusy) setPendingDeleteConversationId(null) }}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  )
}

export default App
