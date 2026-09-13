import { useEffect, useMemo, useRef, useState } from 'react'
import { useCodingSession } from '../hooks/useCodingSession.js'
import { codingActive, codingContext, createCodingFolder } from '../lib/codingSessions.js'
import { readModelToolCapability } from '../lib/toolCalling.js'
import { changeRequest } from '../lib/changeReviews.js'
import { appStorage } from '../lib/appStorage.js'
import { AssistantMarkdown } from '../lib/markdown.jsx'
import { FolderPicker } from './WorkspaceView.jsx'
import { ConversationContext } from '../components/context/ContextEditors.jsx'
import { Button } from '../components/ui/Button.jsx'
import { ConfirmDialog } from '../components/ui/ConfirmDialog.jsx'
import { CamelidMark } from '../components/ui/CamelidMark.jsx'
import { IconApi, IconBolt, IconChat, IconCheck, IconChevronRight, IconCpu, IconFile, IconFolder, IconHistory, IconPlus, IconReceipt, IconSearch, IconSend, IconShield, IconSidebar, IconStop, IconTrash } from '../components/ui/icons.jsx'

const labels = { running: 'Working', queued: 'Queued', working: 'Working', waiting_approval: 'Needs approval', paused: 'Paused', stopping: 'Stopping', completed: 'Completed', done: 'Done', failed: 'Failed', cancelled: 'Stopped', interrupted: 'Interrupted' }
const label = value => labels[value] || value || 'Ready'
const agentName = agent => agent.id === 'lead' ? 'Lead' : agent.id.replaceAll('-', ' ').replace(/^./, c => c.toUpperCase())
const relativeName = path => path?.split(/[\\/]/).at(-1) || path
const ignore = promise => promise?.catch(() => {})
const FileDiff = ({ diff }) => <pre className="coding-diff">{String(diff || '').split('\n').map((line, index) => <span key={index} className={line.startsWith('+') ? 'is-added' : line.startsWith('-') ? 'is-removed' : ''}>{line}{'\n'}</span>)}</pre>

export default function CodingWorkspace({ apiBase, runtime, selectedModel, capabilities, projects, chatContext, contextSources, updateChatContext, globalPrompt, setTab, onActivity, active }) {
  const coding = useCodingSession(apiBase, runtime?.active_model_id, runtime?.loaded_now)
  const { snapshot, busy, connection, selectedId } = coding
  const [workspace, setWorkspace] = useState(() => appStorage.getItem('camelid.codingWorkspace') || '')
  const [goal, setGoal] = useState('')
  const [allowCommands, setAllowCommands] = useState(false)
  const [folderOpen, setFolderOpen] = useState(false)
  const [sideOpen, setSideOpen] = useState(true)
  const [panel, setPanel] = useState('agents')
  const [agentId, setAgentId] = useState('lead')
  const [reviewId, setReviewId] = useState(null)
  const [review, setReview] = useState(null)
  const [reviewError, setReviewError] = useState('')
  const [reviewBusy, setReviewBusy] = useState(false)
  const [reviewOverrides, setReviewOverrides] = useState({})
  const [confirmRemove, setConfirmRemove] = useState(false)
  const bottom = useRef(null)
  const userAway = useRef(false)
  const running = codingActive(snapshot?.phase)
  const capability = readModelToolCapability(capabilities, selectedModel, runtime)
  const ready = capability.capable && runtime?.loaded_now && runtime?.generation_ready && coding.toolCapableModel === runtime.active_model_id
  const agents = Object.values(snapshot?.agents || {}).sort((a, b) => a.id === 'lead' ? -1 : b.id === 'lead' ? 1 : a.id.localeCompare(b.id))
  const selectedAgent = agents.find(a => a.id === agentId) || agents[0]
  const reviews = (snapshot?.reviews || []).map(r => reviewOverrides[r.id] || r)
  const pendingReview = snapshot?.approval?.detail?.review
  const approvalIsSelected = snapshot?.approval && (!reviewId || pendingReview?.id === reviewId)
  const working = agents.filter(a => ['working', 'queued', 'waiting_approval'].includes(a.status))
  const canSend = Boolean(goal.trim() && !running && !busy && ready && (selectedId ? snapshot : workspace.trim()))
  const project = projects.find(p => p.id === (snapshot?.config.project_id || chatContext?.project_id))
  const selectedEvents = useMemo(() => (snapshot?.events || []).filter(e => e.agent_id === selectedAgent?.id && e.kind !== 'model.timing').slice(-20), [snapshot?.events, selectedAgent?.id])
  useEffect(() => { onActivity?.(snapshot ? { id: snapshot.id, title: snapshot.title, phase: snapshot.phase } : null) }, [snapshot?.id, snapshot?.title, snapshot?.phase, onActivity])
  useEffect(() => { setReviewId(null); setAgentId('lead'); setReviewOverrides({}); setReview(null) }, [selectedId])
  useEffect(() => {
    if (!active || running || !snapshot?.reviews?.length) return
    const controller = new AbortController()
    const ids = new Set(snapshot.reviews.map(review => review.id))
    changeRequest(apiBase, '', { signal: controller.signal }).then(data => {
      setReviewOverrides(current => ({ ...current, ...Object.fromEntries((data.reviews || []).filter(review => ids.has(review.id)).map(review => [review.id, review])) }))
    }).catch(error => { if (error.name !== 'AbortError') setReviewError(error.message) })
    return () => controller.abort()
  }, [apiBase, selectedId, running, active, snapshot?.reviews?.length])
  useEffect(() => {
    if (active && !userAway.current) bottom.current?.scrollIntoView({ block: 'end' })
  }, [snapshot?.turns?.length, snapshot?.agents?.lead?.output, snapshot?.approval?.id, active])
  useEffect(() => {
    setReviewError(''); setReview(null)
    if (!reviewId) return
    const controller = new AbortController()
    changeRequest(apiBase, '/' + encodeURIComponent(reviewId), { signal: controller.signal }).then(value => { setReview(value); setReviewOverrides(current => ({ ...current, [value.id]: value })) }).catch(e => { if (e.name !== 'AbortError') setReviewError(e.message) })
    return () => controller.abort()
  }, [apiBase, reviewId, snapshot?.approval?.id, snapshot?.phase])
  const submit = async () => {
    if (!canSend) return
    const options = { workspace: workspace.trim(), model_id: runtime.active_model_id, project_id: chatContext?.project_id || '', ...codingContext(contextSources), allow_commands: allowCommands }
    try { await coding.send(goal.trim(), options); setGoal(''); userAway.current = false; appStorage.setItem('camelid.codingWorkspace', workspace.trim()) } catch { /* hook presents the error */ }
  }
  const decide = async approved => { try { await coding.decide(approved); setReviewId(null) } catch { /* hook presents the error */ } }
  const undo = async () => {
    setReviewBusy(true); setReviewError('')
    try { const next = await changeRequest(apiBase, '/' + encodeURIComponent(reviewId) + '/undo', { method: 'POST' }); setReview(next); setReviewOverrides(current => ({ ...current, [next.id]: next })) }
    catch (e) { setReviewError(e.message) }
    finally { setReviewBusy(false) }
  }
  const decisionButtons = <div className="coding-actions"><Button variant="primary" size="sm" disabled={busy || Boolean(coding.decidingId) || connection !== 'connected' || snapshot?.phase === 'paused'} onClick={() => decide(true)}>{pendingReview ? 'Approve & apply' : 'Allow command once'}</Button><Button variant="outline" size="sm" disabled={busy || Boolean(coding.decidingId) || connection !== 'connected' || snapshot?.phase === 'paused'} onClick={() => decide(false)}>Deny</Button></div>
  const renderApproval = () => snapshot?.approval && <section className="coding-approval" aria-label="Pending agent approval">
    <div className="coding-row"><IconShield size={17} /><strong>{pendingReview ? 'Review file change' : 'Review command'}</strong></div>
    {pendingReview ? <><p className="coding-filepath">{pendingReview.path}</p><FileDiff diff={pendingReview.diff} /><p className="coding-muted">Applies this exact file version. Its original is saved for Undo.</p></> : <><pre>{snapshot.approval.detail.command}</pre><p className="coding-filepath">{snapshot.approval.detail.workspace}</p><p className="coding-muted">{snapshot.approval.detail.execution} Timeout: {snapshot.approval.detail.timeout_seconds} seconds.</p></>}
    {decisionButtons}
  </section>
  return <section className={`coding-workspace ${sideOpen ? '' : 'coding-workspace--wide'}`} aria-label="Agentic coding workspace">
    <header className="coding-toolbar">
      <div className="coding-title"><IconApi size={18} /><strong>{snapshot?.title || 'New coding session'}</strong>{snapshot && <span className={'coding-status is-' + snapshot.phase}>{label(snapshot.phase)}</span>}</div>
      <div className="coding-actions">
        <label className="coding-session-picker"><span className="sr-only">Saved coding sessions</span><IconHistory size={15} /><select aria-label="Saved coding sessions" value={selectedId} disabled={busy || running} onChange={e => coding.select(e.target.value)}><option value="">New session</option>{coding.sessions.map(s => <option value={s.id} key={s.id}>{s.title} · {label(s.phase)}</option>)}</select></label>
        <Button size="sm" variant="ghost" icon={<IconPlus size={16} />} aria-label="New coding session" disabled={busy || running} onClick={() => { coding.select(''); setGoal('') }} />
        <Button size="sm" variant="ghost" icon={<IconSidebar size={16} />} aria-label="Toggle agent sidebar" aria-expanded={sideOpen} aria-controls="coding-team" onClick={() => setSideOpen(v => !v)} />
      </div>
    </header>
    {(coding.error || snapshot?.error || ['reconnecting','disconnected'].includes(connection)) && <div className="coding-alert" role="status"><span>{coding.error || snapshot?.error || 'Connection interrupted. Reconnecting to the server-owned run; no actions will be replayed.'}</span><Button variant="ghost" size="sm" onClick={() => ignore(coding.retry())} disabled={busy}>Refresh</Button></div>}
    <div className="coding-body">
      <div className="coding-conversation" onScroll={event => { const e = event.currentTarget; userAway.current = e.scrollHeight - e.scrollTop - e.clientHeight > 100 }}>
        {reviewId ? <section className="coding-review" aria-label="Selected coding file review">
          <Button size="sm" variant="ghost" icon={<IconChat size={16} />} onClick={() => setReviewId(null)}>Conversation</Button>
          {reviewError && <p role="alert">{reviewError}</p>}
          {!review && !reviewError && <p role="status">Loading file review…</p>}
          {review && <><div className="coding-row"><h2>{relativeName(review.path)}</h2><span className="coding-status">{review.status}</span></div><p className="coding-filepath">{review.path}</p><FileDiff diff={review.diff} />
            <details><summary>Complete before and after</summary><h3>Before</h3><pre>{review.before ?? '(New file)'}</pre><h3>After</h3><pre>{review.after}</pre></details>
            {approvalIsSelected && pendingReview?.id === review.id && decisionButtons}
            {review.status === 'applied' && <div className="coding-approval"><p>Undo restores the saved original only if the file has not changed since application.</p><Button variant="outline" size="sm" disabled={reviewBusy || running} onClick={undo}>Undo change</Button>{running && <p className="coding-muted">Wait for the run to finish before undoing its changes.</p>}</div>}
            {review.status === 'pending' && !approvalIsSelected && <p className="coding-muted">This saved review is not a pending action in the current run. Open Changes to review it separately.</p>}
          </>}
        </section> : <>
          {!snapshot && <div className="coding-empty"><CamelidMark size={38} /><h2>Put Camelid to work in your project</h2><p>Ask for a fix, a feature, or a refactor. Follow the lead and its helpers as they inspect files, propose changes, and check the result.</p></div>}
          {snapshot?.turns.map((turn, index) => <article className="coding-turn" key={index}>
            <p className="coding-user">{turn.user}</p>
            {(turn.assistant || index === snapshot.turns.length - 1) && <div className="coding-answer"><div className="coding-byline"><CamelidMark size={20} /><strong>Camelid</strong><span>Lead agent</span></div>{turn.assistant ? <AssistantMarkdown content={turn.assistant} /> : index === snapshot.turns.length - 1 && snapshot.agents.lead?.output ? <pre className="coding-live">{snapshot.agents.lead.output}</pre> : <p className="coding-muted">{running ? 'Inspecting the project and choosing the next action…' : label(turn.outcome)}</p>}</div>}
          </article>)}
          {snapshot?.plan?.length > 0 && <ol className="coding-plan" aria-label="Agent task plan">{snapshot.plan.map((step, i) => <li key={i} className={'is-' + step.status}><span>{step.status === 'done' ? <IconCheck size={14} /> : i + 1}</span>{step.text}</li>)}</ol>}
          {reviews.length > 0 && <section className="coding-proposals" aria-label="Proposed changes"><div className="coding-row"><IconReceipt size={17} /><strong>File changes</strong><span className="coding-muted">{reviews.length}</span></div>{reviews.map(r => <button type="button" key={r.id} onClick={() => setReviewId(r.id)}><IconFile size={16} /><span>{r.path}</span><small>{r.status}</small><IconChevronRight size={14} /></button>)}</section>}
          {renderApproval()}
          {running && <div className="coding-runline" role="status"><span className="coding-dot" /><span>{snapshot.phase === 'paused' ? 'Paused before the next action. The current model request or command may finish.' : snapshot.phase === 'stopping' ? 'Stopping model requests and tools…' : snapshot.agents.lead?.action || 'Working'}</span><div className="coding-actions"><Button size="sm" variant="ghost" disabled={busy || snapshot.phase === 'stopping'} onClick={() => ignore(coding.control(snapshot.phase === 'paused' ? 'resume' : 'pause'))}>{snapshot.phase === 'paused' ? 'Resume' : 'Pause'}</Button><Button size="sm" variant="ghost" icon={<IconStop size={14} />} disabled={busy || snapshot.phase === 'stopping'} onClick={() => ignore(coding.control('stop'))}>Stop</Button></div></div>}
          {snapshot && !running && <div className="coding-runline"><span>{label(snapshot.phase)} · saved on this machine</span><Button size="sm" variant="ghost" icon={<IconTrash size={14} />} disabled={busy} onClick={() => setConfirmRemove(true)}>Remove session</Button></div>}
          <div ref={bottom} />
        </>}
      </div>
      <aside id="coding-team" className="coding-team" hidden={!sideOpen} aria-label="Agent assignments">
        <div className="coding-tabs" role="group" aria-label="Coding sidebar"><button type="button" aria-pressed={panel === 'agents'} onClick={() => setPanel('agents')}>Agents <span>{agents.length}</span></button><button type="button" aria-pressed={panel === 'changes'} onClick={() => setPanel('changes')}>Changes <span>{reviews.length}</span></button></div>
        {panel === 'agents' ? <>
          <p className="coding-team-summary">{agents.length ? `${working.length} active · ${agents.filter(a => a.status === 'done').length} completed` : 'Your team appears when a run starts'}</p>
          <div className="coding-agent-list">{agents.map(a => <button type="button" className="coding-agent" key={a.id} aria-pressed={selectedAgent?.id === a.id} onClick={() => setAgentId(a.id)}><span className="coding-row">{a.id === 'lead' ? <IconApi size={17} /> : <IconSearch size={17} />}<strong>{agentName(a)}</strong><span className={'coding-status is-' + a.status}>{label(a.status)}</span></span><span>{a.goal}</span><small>{a.action}</small></button>)}</div>
          {selectedAgent && <section className="coding-agent-detail"><div className="cx-kicker">Selected agent</div><h3>{agentName(selectedAgent)}</h3><p>{selectedAgent.parent_id ? 'Read-only helper · reports to Lead' : 'Lead · edits and commands require approval'}</p>{selectedAgent.files.map(f => <div className="coding-agent-file" key={f}><IconFile size={13} /><span>{f}</span></div>)}
            {selectedAgent.id !== 'lead' && selectedAgent.output && <details open={selectedAgent.status === 'done'}><summary>Findings</summary><pre>{selectedAgent.output}</pre></details>}
            <details><summary>Activity ({selectedEvents.length})</summary><ol className="coding-event-list">{selectedEvents.map(e => <li key={e.seq}><small>{new Date(e.time).toLocaleTimeString()}</small><strong>{e.kind.replaceAll('.', ' ')}</strong><pre>{e.detail?.content || e.detail?.detail || e.detail?.message || JSON.stringify(e.detail)}</pre></li>)}</ol></details>
          </section>}
          <div className="coding-team-footer"><IconCpu size={15} /><span>One local model · up to 2 read-only helpers</span></div>
        </> : <div className="coding-review-list">{reviews.length ? reviews.map(r => <button type="button" key={r.id} onClick={() => setReviewId(r.id)}><IconFile size={15} /><span>{r.path}<small>{r.status}</small></span></button>) : <p>File changes appear here when Lead proposes an edit.</p>}</div>}
      </aside>
    </div>
    <div className="coding-composer-area">
      {!snapshot && <div className="coding-setup">
        <div className="coding-folder-field"><label htmlFor="coding-folder">Project folder</label><div className="coding-row"><input id="coding-folder" value={workspace} onChange={e => setWorkspace(e.target.value)} placeholder="Choose a local folder" disabled={busy} /><Button size="sm" variant="outline" icon={<IconFolder size={15} />} disabled={busy} onClick={() => setFolderOpen(true)}>Browse</Button></div></div>
        <label className="coding-command-choice"><input type="checkbox" checked={allowCommands} onChange={e => setAllowCommands(e.target.checked)} disabled={busy} /><span>Allow command requests<small>Every command asks first and runs with your account permissions. Command changes are not covered by Undo.</small></span></label>
      </div>}
      {!snapshot ? <ConversationContext compact={false} context={chatContext} projects={projects} sources={contextSources} globalPrompt={globalPrompt || ''} onSave={updateChatContext} onManageProjects={() => setTab('projects')} busy={busy} /> : <div className="coding-context"><IconFolder size={15} /><strong>{project?.name || relativeName(snapshot.config.workspace)}</strong><span>{snapshot.config.workspace}</span></div>}
      {!ready && <p className="coding-readiness" role="status">{capability.reason || 'Code needs an exact tool-capable model artifact with a certified digest.'} <button type="button" onClick={() => setTab('library')}>Open Models</button></p>}
      <form className="coding-composer" onSubmit={e => { e.preventDefault(); submit() }}>
        <textarea aria-label="Message coding agents" value={goal} maxLength={16000} onChange={e => setGoal(e.target.value)} placeholder={running ? 'Wait for this run, or pause or stop it above…' : snapshot ? 'Give Lead a follow-up…' : 'What should Camelid build or fix?'} disabled={busy || running} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit() } }} rows={2} />
        <div className="coding-composer-tools"><span><span className="coding-dot" />{snapshot?.config.model_id || selectedModel?.name || 'No model'}</span><details><summary><IconBolt size={14} />Project tools</summary><p>Read files · search · plan · reviewed edits · read-only helpers{(snapshot?.config.allow_commands ?? allowCommands) ? ' · approved commands' : ''}</p></details><span><IconShield size={14} />Review changes</span><Button type="submit" size="sm" variant="primary" icon={<IconSend size={17} />} aria-label={snapshot ? 'Send coding follow-up' : 'Start coding'} disabled={!canSend} /></div>
      </form>
    </div>
    {folderOpen && <FolderPicker apiBase={apiBase} initialPath={workspace || null} onCreate={(parent, name) => createCodingFolder(apiBase, parent, name)} onPick={value => { setWorkspace(value); setFolderOpen(false) }} onClose={() => setFolderOpen(false)} />}
    <ConfirmDialog open={confirmRemove} title="Remove this coding session?" detail="This removes the saved conversation and agent activity. Reviewed file changes and their Undo history remain in Changes." confirmLabel="Remove session" onCancel={() => setConfirmRemove(false)} onConfirm={async () => { try { await coding.remove(); setConfirmRemove(false) } catch { /* visible hook error */ } }} />
  </section>
}
