import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
import {
  cancelWorkspaceSession,
  createWorkspaceSession,
  decideWorkspaceApproval,
  reduceWorkspaceEvent,
  WORKSPACE_IDLE_STATE,
  workspaceEndpoint,
} from '../lib/workspaceAgent'
import { Button } from '../components/ui/Button'
import { Modal } from '../components/ui/Modal'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import {
  IconBolt, IconCheckCircle, IconClose, IconEdit, IconError, IconPlay, IconSearch, IconStop,
} from '../components/ui/icons'

const PHASE_LABEL = {
  idle: 'Ready',
  starting: 'Starting',
  running: 'Running',
  awaiting_approval: 'Approval needed',
  finished: 'Complete',
  aborted: 'Stopped',
  cancelled: 'Stopped',
  step_capped: 'Step limit reached',
  repeated: 'No progress',
  driver_error: 'Model error',
  error: 'Error',
}

function initialWorkspaceState() {
  return { ...WORKSPACE_IDLE_STATE, events: [] }
}

function eventKey(event, index) {
  return `${event.sequence || 'local'}-${event.event}-${index}`
}

function ActivityRow({ event }) {
  const kind = event.event
  if (kind === 'session.started') {
    return <li className="workspace-event workspace-event--system"><IconPlay size={16} /><div><strong>Session started</strong><span>{event.model_id}</span></div></li>
  }
  if (kind === 'tool.call') {
    return <li className="workspace-event workspace-event--tool"><IconBolt size={16} /><div><strong>Tool requested</strong><code>{event.detail}</code></div></li>
  }
  if (kind === 'approval.required') {
    return <li className="workspace-event workspace-event--approval"><IconEdit size={16} /><div><strong>Waiting for approval</strong><span>{event.tool}</span></div></li>
  }
  if (kind === 'tool.result') {
    const failed = event.outcome === 'error'
    return <li className={`workspace-event ${failed ? 'workspace-event--error' : 'workspace-event--result'}`}>{failed ? <IconError size={16} /> : <IconCheckCircle size={16} />}<div><strong>{failed ? 'Tool failed' : 'Tool complete'}</strong><span>{event.tool}</span><pre>{event.content}</pre></div></li>
  }
  if (kind === 'model.live' || kind === 'model.answer') {
    return <li className={`workspace-event workspace-event--model ${kind === 'model.live' ? 'is-live' : ''}`}><IconBolt size={16} /><div><strong>{kind === 'model.live' ? 'Model working' : 'Camelid'}</strong><pre>{event.content}</pre></div></li>
  }
  if (kind === 'session.finished') {
    return <li className="workspace-event workspace-event--system"><IconCheckCircle size={16} /><div><strong>Session finished</strong><span>{PHASE_LABEL[event.outcome] || event.outcome}</span></div></li>
  }
  if (kind === 'session.error') {
    return <li className="workspace-event workspace-event--error"><IconError size={16} /><div><strong>Session error</strong><span>{event.message}</span></div></li>
  }
  return <li className="workspace-event workspace-event--system"><IconBolt size={16} /><div><strong>Workspace</strong><span>{event.content || kind}</span></div></li>
}

function ApprovalDialog({ approval, busy, onDecision }) {
  return (
    <Modal
      open={Boolean(approval)}
      onClose={() => { if (!busy) onDecision('deny') }}
      title="Review file change"
      labelledById="workspace-approval-title"
      size="md"
      className="workspace-approval-modal"
      overlayClassName="workspace-approval-overlay"
      footer={
        <div className="workspace-approval__actions">
          <Button variant="ghost" onClick={() => onDecision('abort')} disabled={busy}>Stop session</Button>
          <span className="workspace-approval__spacer" />
          <Button variant="ghost" onClick={() => onDecision('deny')} disabled={busy}>Deny</Button>
          <Button variant="outline" onClick={() => onDecision('always_tool')} disabled={busy}>Always allow {approval?.tool}</Button>
          <Button variant="primary" onClick={() => onDecision('allow_once')} loading={busy}>Allow once</Button>
        </div>
      }
    >
      <div className="workspace-approval">
        <div className="workspace-approval__risk"><IconEdit size={18} /><span>{approval?.risk} action</span></div>
        <pre>{approval?.detail}</pre>
        <p>This action has already been resolved against the selected workspace. The browser cannot change its target.</p>
      </div>
    </Modal>
  )
}

export default function WorkspaceView({ apiBase, capabilities, selectedModel, runtime, setTab }) {
  const [workspacePath, setWorkspacePath] = useState(() => window.localStorage.getItem('camelid.workspacePath') || '')
  const [goal, setGoal] = useState('')
  const [session, setSession] = useState(null)
  const [state, dispatch] = useReducer(reduceWorkspaceEvent, undefined, initialWorkspaceState)
  const [approvalBusy, setApprovalBusy] = useState(false)
  const eventSourceRef = useRef(null)
  const intentionalClosuresRef = useRef(new WeakSet())
  const timelineRef = useRef(null)

  const compatibility = useMemo(
    () => findCompatibilityHint(capabilities, selectedModel, null),
    [capabilities, selectedModel],
  )
  const target = compatibility?.target || null
  const toolCapable = Boolean(compatibility?.exact && target?.tool_capable && String(target.status || '').startsWith('supported'))
  const runtimeReady = runtime?.status === 'online' && runtime?.loaded_now && runtime?.generation_ready
  const running = ['starting', 'running', 'awaiting_approval'].includes(state.phase)
  const canStart = Boolean(workspacePath.trim() && goal.trim() && toolCapable && runtimeReady && !running)

  useEffect(() => {
    if (workspacePath) window.localStorage.setItem('camelid.workspacePath', workspacePath)
  }, [workspacePath])

  useEffect(() => {
    timelineRef.current?.scrollTo({ top: timelineRef.current.scrollHeight, behavior: 'smooth' })
  }, [state.events.length, state.phase])

  useEffect(() => () => {
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
  }, [])

  const openEventStream = (created) => {
    const url = workspaceEndpoint(apiBase, `/${encodeURIComponent(created.id)}/events`)
    const source = new EventSource(url)
    eventSourceRef.current = source
    source.addEventListener('workspace', (message) => {
      try {
        const envelope = JSON.parse(message.data)
        dispatch(envelope)
        if (['session.finished', 'session.error'].includes(envelope.event)) {
          intentionalClosuresRef.current.add(source)
          source.close()
          eventSourceRef.current = null
        }
      } catch {
        dispatch({ event: 'session.error', message: 'Camelid returned an unreadable Workspace event.' })
        intentionalClosuresRef.current.add(source)
        source.close()
      }
    })
    source.onerror = () => {
      if (intentionalClosuresRef.current.has(source)) return
      if (eventSourceRef.current !== source) return
      dispatch({ event: 'session.error', message: 'The Workspace event stream disconnected.' })
      source.close()
      eventSourceRef.current = null
    }
  }

  const start = async () => {
    if (!canStart) return
    dispatch({ event: 'session.starting' })
    try {
      const created = await createWorkspaceSession(apiBase, {
        workspace: workspacePath.trim(),
        goal: goal.trim(),
        max_steps: 12,
        max_tokens: 800,
        temperature: 0,
      })
      setSession(created)
      openEventStream(created)
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    }
  }

  const stop = async () => {
    if (!session) return
    try {
      await cancelWorkspaceSession(apiBase, session.id)
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      if (eventSourceRef.current) {
        intentionalClosuresRef.current.add(eventSourceRef.current)
        eventSourceRef.current.close()
      }
      eventSourceRef.current = null
      dispatch({ event: 'session.finished', outcome: 'cancelled' })
    }
  }

  const decide = async (decision) => {
    const approval = state.pendingApproval
    if (!session || !approval || approvalBusy) return
    setApprovalBusy(true)
    try {
      if (decision === 'abort') {
        await stop()
      } else {
        await decideWorkspaceApproval(apiBase, session.id, approval.approval_id, decision)
        dispatch({ event: 'approval.resolved' })
        dispatch({ event: 'session.notice', content: decision === 'deny' ? 'Action denied.' : 'Action approved.' })
      }
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      setApprovalBusy(false)
    }
  }

  const reset = () => {
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
    eventSourceRef.current = null
    setSession(null)
    dispatch({ event: 'session.reset' })
  }

  const statusLabel = PHASE_LABEL[state.phase] || state.phase

  return (
    <div className="workspace-view">
      <section className="workspace-setup" aria-labelledby="workspace-heading">
        <div className="workspace-setup__heading">
          <div>
            <p className="workspace-kicker">Local file workspace</p>
            <h2 id="workspace-heading">Give Camelid a bounded task</h2>
          </div>
          <span className={`workspace-status is-${state.phase}`}>{statusLabel}</span>
        </div>

        <div className="workspace-model-line">
          <span>Active model</span>
          <strong>{selectedModel?.name || runtime?.active_model_id || 'No model loaded'}</strong>
          <EvidenceChip
            status={target?.status || ''}
            state={toolCapable ? 'supported' : 'unsupported'}
            label={toolCapable ? 'Agent evaluated' : 'Not agent evaluated'}
            source={{ rowId: target?.id, note: toolCapable ? 'This exact row has a committed agent-eval PASS.' : 'Workspace requires an exact row with tool_capable=true.' }}
            size="sm"
          />
          {!runtimeReady && <button type="button" className="workspace-link" onClick={() => setTab('library')}>Choose a ready model</button>}
        </div>

        {!toolCapable && (
          <div className="workspace-blocked" role="status">
            <IconError size={18} />
            <span>This exact model has not passed Camelid’s tool-use battery. Chat remains available, but Workspace stays locked.</span>
          </div>
        )}

        <label className="workspace-field">
          <span>Workspace folder</span>
          <input
            value={workspacePath}
            onChange={(event) => setWorkspacePath(event.target.value)}
            placeholder={navigator.platform?.startsWith('Win') ? 'C:\\projects\\example' : '/Users/you/projects/example'}
            disabled={running}
            spellCheck="false"
          />
          <small>Camelid canonicalizes this directory and rejects paths that leave it.</small>
        </label>

        <label className="workspace-field workspace-field--goal">
          <span>Goal</span>
          <textarea
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="Review this folder, find why the tests fail, and propose the smallest repair."
            rows={4}
            disabled={running}
          />
        </label>

        <div className="workspace-setup__actions">
          {running ? (
            <Button variant="outline" onClick={stop}><IconStop size={17} /> Stop</Button>
          ) : (
            <Button variant="primary" onClick={start} disabled={!canStart}><IconPlay size={17} /> Start Workspace</Button>
          )}
          {!running && state.events.length > 0 && <Button variant="ghost" onClick={reset}><IconClose size={17} /> Clear activity</Button>}
          <span>12 steps · read-only tools run automatically · every write asks first</span>
        </div>
      </section>

      <section className="workspace-activity" aria-labelledby="workspace-activity-heading">
        <div className="workspace-activity__header">
          <div>
            <p className="workspace-kicker">Activity</p>
            <h2 id="workspace-activity-heading">Session timeline</h2>
          </div>
          {session?.workspace && <code title={session.workspace}>{session.workspace}</code>}
        </div>
        <div className="workspace-activity__scroll" ref={timelineRef}>
          {state.events.length === 0 ? (
            <div className="workspace-empty">
              <IconSearch size={24} />
              <strong>No activity yet</strong>
              <span>Start a task to see every model step, file operation, and approval in order.</span>
            </div>
          ) : (
            <ol className="workspace-timeline">
              {state.events.map((event, index) => <ActivityRow key={eventKey(event, index)} event={event} />)}
            </ol>
          )}
        </div>
      </section>

      <ApprovalDialog approval={state.pendingApproval} busy={approvalBusy} onDecision={decide} />
    </div>
  )
}
