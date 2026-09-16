import { useEffect, useRef, useState } from 'react'
import { Button } from '../ui/Button.jsx'
import { ConfirmDialog } from '../ui/ConfirmDialog.jsx'
import { appStorage } from '../../lib/appStorage.js'

const ignore = promise => promise?.catch(() => {})
const escapedJson = value => JSON.stringify(value).replaceAll('<', '\\u003c')

export function CodingProjectPanel({ panel, coding, settings, setSettings, running }) {
  const { snapshot, executionEngine, busy } = coding
  const [draft, setDraft] = useState(settings)
  const [workflowName, setWorkflowName] = useState('')
  const [notes, setNotes] = useState('')
  const [saved, setSaved] = useState('')
  const [preview, setPreview] = useState(null)
  const [previewError, setPreviewError] = useState('')
  const [loading, setLoading] = useState(false)
  const [server, setServer] = useState(null)
  const [serverBusy, setServerBusy] = useState(false)
  const [serverError, setServerError] = useState('')
  const selectedSession = useRef(snapshot?.id)
  selectedSession.current = snapshot?.id
  const [restore, setRestore] = useState(null)
  const frame = useRef(null)
  const projectKey = JSON.stringify(snapshot?.config.project || settings)
  useEffect(() => { setDraft(JSON.parse(projectKey)) }, [projectKey])
  useEffect(() => { setPreview(null); setPreviewError(''); setSaved('') }, [snapshot?.id])
  useEffect(() => { setServer(null); setServerError(''); setServerBusy(false) }, [snapshot?.id])
  useEffect(() => {
    if (panel !== 'preview' || !snapshot?.id || !coding.previewServerStatus) return
    const controller = new AbortController()
    coding.previewServerStatus(controller.signal).then(status => { if (!controller.signal.aborted) setServer(status) }).catch(error => {
      if (!controller.signal.aborted) setServerError(error.message)
    })
    return () => controller.abort()
  }, [panel, snapshot?.id, snapshot?.seq])
  const manageServer = async action => {
    const session = snapshot?.id
    setServerBusy(true); setServerError('')
    try {
      const status = await coding.managePreview(action, draft.preview_entry || '')
      if (selectedSession.current === session) setServer(status)
    } catch (error) {
      if (selectedSession.current === session) {
        setServerError(error.message)
        coding.previewServerStatus().then(status => { if (selectedSession.current === session) setServer(status) }).catch(() => {})
      }
    } finally { if (selectedSession.current === session) setServerBusy(false) }
  }
  const storageKey = `camelid.preview.${snapshot?.config.project?.engine_id || 'engine'}.${snapshot?.config.workspace || ''}`
  useEffect(() => {
    const receive = event => {
      if (event.source !== frame.current?.contentWindow || event.data?.type !== 'camelid-preview-storage') return
      const data = event.data.values
      if (!data || typeof data !== 'object' || Array.isArray(data) || Object.keys(data).length > 128 || Object.values(data).some(v => typeof v !== 'string')) return
      const text = JSON.stringify(data)
      if (text.length <= 256 * 1024) appStorage.setItem(storageKey, text)
    }
    window.addEventListener('message', receive)
    return () => window.removeEventListener('message', receive)
  }, [storageKey])
  const loadPreview = async () => {
    setLoading(true); setPreviewError('')
    try {
      const data = await coding.loadPreview()
      let stored = {}
      try { stored = JSON.parse(appStorage.getItem(storageKey) || '{}') } catch { /* use a fresh preview store */ }
      // Opaque iframe origins cannot use native localStorage. Supply only this
      // project's bounded store; never expose the engine's cookies or storage.
      const bridge = `<script>(()=>{const values=Object.assign(Object.create(null),${escapedJson(stored)});const publish=()=>parent.postMessage({type:'camelid-preview-storage',values:{...values}},'*');Object.defineProperty(window,'localStorage',{value:{getItem:k=>Object.hasOwn(values,String(k))?values[String(k)]:null,setItem:(k,v)=>{k=String(k);v=String(v);if(k.length+v.length>262144||Object.keys(values).length>=128&&!Object.hasOwn(values,k))throw new DOMException('Preview storage limit','QuotaExceededError');const next={...values,[k]:v};if(JSON.stringify(next).length>262144)throw new DOMException('Preview storage limit','QuotaExceededError');values[k]=v;publish()},removeItem:k=>{delete values[String(k)];publish()},clear:()=>{for(const k of Object.keys(values))delete values[k];publish()},key:i=>Object.keys(values)[i]??null,get length(){return Object.keys(values).length}}});})();</script>`
      setPreview({ ...data, html: bridge + data.html })
    } catch (error) { setPreviewError(error.message) }
    finally { setLoading(false) }
  }
  if (panel === 'preview') return <section className="coding-project-panel" aria-label="Project preview">
    <h3>Test in Chrome</h3>
    <p>Start a local server for your HTML, CSS and JavaScript. It stays available after the task finishes, until you stop it or close Camelid.</p>
    <div className="coding-row">
      <Button size="sm" disabled={!snapshot || serverBusy} onClick={() => manageServer('start')}>{server?.running ? 'Preview running' : 'Start preview server'}</Button>
      <Button size="sm" disabled={!snapshot || serverBusy} onClick={() => manageServer('open')}>Open in Chrome</Button>
      <Button size="sm" disabled={!server?.running || serverBusy} onClick={() => manageServer('stop')}>Stop preview server</Button>
    </div>
    {server?.running && <p><a href={server.url} target="_blank" rel="noopener noreferrer">{server.url}</a><br /><span className="coding-muted">Reload Chrome after file changes. Opening the site does not record a passing test.</span></p>}
    {serverError && <p role="alert">{serverError}</p>}
    <h3>Project preview</h3><p>Open the static HTML page with its local CSS and JavaScript. Preview storage is saved separately for this project.</p>
    <Button size="sm" disabled={!snapshot || loading || running} onClick={loadPreview}>{loading ? 'Loading…' : preview ? 'Refresh preview' : 'Open preview'}</Button>
    {running && <p>Wait for the current edits to settle before opening a preview.</p>}
    {previewError && <p role="alert">{previewError}</p>}
    {preview && <><iframe ref={frame} title="Isolated project preview" sandbox="allow-scripts" srcDoc={preview.html} /><p className="coding-muted">{preview.limitations}</p></>}
  </section>
  if (panel === 'checks') return <section className="coding-project-panel" aria-label="Project checks">
    <h3>Verification</h3>
    <p>{snapshot?.reviews?.some(r => r.status === 'applied') ? 'File changes recorded.' : 'No applied file changes yet.'} Check results below come from commands run on the selected engine.</p>
    <Button size="sm" disabled={!snapshot?.config.allow_commands || running || busy} onClick={() => ignore(coding.send('Run verify_project on the current project. Report exactly which check ran and its observed result. Do not claim browser interactions unless they were tested.', {}))}>Run project checks</Button>
    {!snapshot?.config.allow_commands && <p>Enable command approval when starting a session to run checks.</p>}
    {!(snapshot?.checks?.length) && <p className="coding-muted">Not verified yet. Opening a page or marking a plan done does not create a passing check.</p>}
    {[...(snapshot?.checks || [])].reverse().map(check => <article className={`coding-check is-${check.status}`} key={check.id}>
      <div className="coding-row"><strong>{check.status === 'passed' ? 'Check passed' : check.status === 'failed' ? 'Needs fixing' : check.status === 'stale' ? 'Changed since check' : 'Check not run'}</strong><small>{new Date(check.time).toLocaleTimeString()}</small></div>
      <p>{Object.keys(check.files || {}).length} file/preview versions recorded</p>
      <details><summary>Command and output</summary><pre>{check.command}{'\n\n'}{check.output}</pre></details>
    </article>)}
  </section>
  return <section className="coding-project-panel" aria-label="Coding project settings">
    <h3>Execution machine</h3>
    <label>Run on<select aria-label="Execution machine" value={executionEngine?.id || ''} disabled><option value={executionEngine?.id || ''}>{executionEngine?.name || 'Connected engine'}</option></select></label>
    <p>Files, agents, commands, and checks stay on this engine. There is no fallback to the browser's machine.</p>
    <p className="coding-muted">One model request at a time. Build tools default to one job. Commands have a 120-second timeout and run with this account's permissions.</p>
    <form onSubmit={async event => {
      event.preventDefault(); setSaved('')
      const next = { ...draft, engine_id: executionEngine?.id || '' }
      if (!snapshot) { setSettings(next); setSaved('Settings selected for the new session.'); return }
      try { await coding.projectAction({ action: 'settings', run_id: snapshot.run_id, settings: next }); setSaved('Project settings saved.') } catch { /* hook displays errors */ }
    }}>
      <label>Verification command<input aria-label="Verification command" value={draft.verification_command || ''} maxLength={4000} disabled={running} placeholder="For example: npm test" onChange={e => setDraft(v => ({ ...v, verification_command: e.target.value }))} /></label>
      <p className="coding-muted">The agent asks before running this exact command. With no command configured, an existing app.js can receive a JavaScript syntax check.</p>
      <label className="coding-project-checkbox"><input aria-label="Browser page-load check" type="checkbox" checked={Boolean(draft.browser_check)} disabled={running} onChange={e => setDraft(v => ({ ...v, browser_check: e.target.checked }))} />Also check the page loads without JavaScript errors</label>
      <p className="coding-muted">Runs an isolated Chromium browser on the engine after approval. Interaction tests belong in your verification command.</p>
      <label>Preview entry<input aria-label="Preview entry" value={draft.preview_entry || ''} disabled={running} placeholder="index.html" onChange={e => setDraft(v => ({ ...v, preview_entry: e.target.value }))} /></label>
      <label>Saved workflow<input aria-label="Saved workflow" value={draft.workflow || ''} disabled={running} placeholder="Optional workflow name" onChange={e => setDraft(v => ({ ...v, workflow: e.target.value }))} /></label>
      <label>Task time budget (minutes)<input aria-label="Task time budget" type="number" min="1" max="120" value={(draft.max_run_seconds || 1800) / 60} disabled={running} onChange={e => setDraft(v => ({ ...v, max_run_seconds: Math.round(Number(e.target.value) * 60) }))} /></label>
      <p className="coding-muted">The budget includes generation, tools, pauses, approvals, and helper waits.</p>
      <Button type="submit" size="sm" disabled={running || busy}>Save project settings</Button>
    </form>
    {saved && <p role="status">{saved}</p>}
    {snapshot && <>
      <h3>Reusable workflow</h3><p>Save instructions for repeating a verified procedure in this project. A current passing check is required.</p>
      <form onSubmit={async event => { event.preventDefault(); try { await coding.projectAction({ action: 'save_workflow', name: workflowName, notes }); setSaved(`Workflow ${workflowName} saved. Select its name above in future sessions.`) } catch { /* hook displays errors */ } }}>
        <label>Name<input aria-label="Workflow name" value={workflowName} maxLength={80} onChange={e => setWorkflowName(e.target.value)} /></label>
        <label>Instructions<textarea aria-label="Workflow instructions" value={notes} maxLength={8000} onChange={e => setNotes(e.target.value)} /></label>
        <Button type="submit" size="sm" disabled={running || busy || !workflowName.trim() || !notes.trim() || !snapshot.checks?.some(c => c.status === 'passed')}>Save verified workflow</Button>
      </form>
      <h3>Task checkpoints</h3><p>Restore a task's file changes together. Later manual edits are protected; command side effects remain outside Undo.</p>
      {[...(snapshot.checkpoints || [])].reverse().map(checkpoint => <article className="coding-check" key={checkpoint.id}>
        <strong>{checkpoint.title}</strong><p>{checkpoint.review_ids.length} changes · {checkpoint.status}</p>
        <Button size="sm" variant="outline" disabled={running || busy || !checkpoint.review_ids.length || checkpoint.status === 'restored'} onClick={() => setRestore(checkpoint)}>Restore checkpoint</Button>
      </article>)}
      <ConfirmDialog open={Boolean(restore)} title="Restore this task checkpoint?" detail={restore ? `Restore the ${restore.review_ids.length} file changes recorded for “${restore.title}”. Files with newer edits will block restoration.` : ''} confirmLabel="Restore checkpoint" onCancel={() => setRestore(null)} onConfirm={async () => { try { await coding.projectAction({ action: 'restore_checkpoint', run_id: snapshot.run_id, checkpoint_id: restore.id }); setRestore(null) } catch { /* hook displays errors */ } }} />
    </>}
  </section>
}
