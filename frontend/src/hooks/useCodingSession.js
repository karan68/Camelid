import { useCallback, useEffect, useRef, useState } from 'react'
import { appStorage } from '../lib/appStorage.js'
import { acceptCodingSnapshot, codingActive, codingEndpoint, codingMessageId, codingRequest } from '../lib/codingSessions.js'

export function useCodingSession(apiBase, activeModelId, loadedNow) {
  const [selectedId, setSelectedId] = useState(() => appStorage.getItem('camelid.codingSession') || '')
  const [snapshot, setSnapshot] = useState(null)
  const [sessions, setSessions] = useState([])
  const [toolCapableModel, setToolCapableModel] = useState(null)
  const [error, setError] = useState('')
  const [connection, setConnection] = useState('loading')
  const [busy, setBusy] = useState(false)
  const [decidingId, setDecidingId] = useState('')
  const mutation = useRef(false)
  const attempt = useRef(null)
  const selected = useRef(selectedId)
  selected.current = selectedId
  const accept = useCallback(next => {
    setSnapshot(current => acceptCodingSnapshot(current, next, selected.current))
    if (next?.id === selected.current) setSessions(current => [{ id: next.id, title: next.title, phase: next.phase, updated_at: next.updated_at, workspace: next.config.workspace }, ...current.filter(s => s.id !== next.id)])
  }, [])
  const refresh = useCallback(async signal => {
    const data = await codingRequest(apiBase, '', { signal })
    if (!Array.isArray(data.sessions)) throw new Error('Coding sessions are unavailable from this engine.')
    setSessions(data.sessions)
    setToolCapableModel(data.tool_capable_model || null)
    return data.sessions
  }, [apiBase])
  useEffect(() => {
    const controller = new AbortController()
    refresh(controller.signal).then(sessions => {
      const active = sessions.find(session => codingActive(session.phase))
      if (active && active.id !== selected.current) {
        appStorage.setItem('camelid.codingSession', active.id)
        selected.current = active.id; setSelectedId(active.id)
      }
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message) })
    return () => controller.abort()
  }, [refresh, activeModelId, loadedNow])
  useEffect(() => {
    setSnapshot(null); setError(''); setDecidingId('')
    if (!selectedId) { setConnection('idle'); return }
    const controller = new AbortController()
    setConnection('loading')
    codingRequest(apiBase, '/' + encodeURIComponent(selectedId), { signal: controller.signal }).then(data => { accept(data); setConnection('connected') }).catch(e => {
      if (e.name !== 'AbortError') { setError(e.message); setConnection('disconnected') }
    })
    return () => controller.abort()
  }, [apiBase, selectedId, accept])
  useEffect(() => {
    if (!selectedId || snapshot?.id !== selectedId || !snapshot.run_id) return
    const source = new EventSource(codingEndpoint(apiBase, '/' + encodeURIComponent(selectedId) + '/events'))
    let terminal = false
    source.addEventListener('coding', event => {
      try {
        const next = JSON.parse(event.data)
        if (next.id !== selected.current) return
        accept(next); setConnection('connected')
        if (!codingActive(next.phase)) { terminal = true; source.close(); refresh().catch(e => setError(e.message)) }
      } catch { setError('Received an unreadable coding update. Reconnecting will restore the current run.'); setConnection('reconnecting') }
    })
    source.onopen = () => setConnection('connected')
    source.onerror = () => { if (!terminal) setConnection('reconnecting') }
    return () => source.close()
  }, [apiBase, selectedId, snapshot?.run_id, accept, refresh])
  useEffect(() => { if (snapshot?.approval?.id !== decidingId) setDecidingId('') }, [snapshot?.approval?.id, decidingId])
  const select = useCallback(id => {
    appStorage.setItem('camelid.codingSession', id)
    selected.current = id; setSelectedId(id); setSnapshot(null)
  }, [])
  const mutate = useCallback(async operation => {
    if (mutation.current) return
    mutation.current = true; setBusy(true); setError('')
    try { return await operation() }
    catch (e) { setError(e.message); throw e }
    finally { mutation.current = false; setBusy(false) }
  }, [])
  const send = async (goal, options) => mutate(async () => {
    const signature = JSON.stringify({ selectedId, goal, ...(selectedId ? {} : options) })
    if (attempt.current?.signature !== signature) attempt.current = { signature, message_id: codingMessageId() }
    const next = selectedId
      ? await codingRequest(apiBase, '/' + encodeURIComponent(selectedId) + '/messages', { method: 'POST', body: { message: goal, message_id: attempt.current.message_id } })
      : await codingRequest(apiBase, '', { method: 'POST', body: { ...options, goal, message_id: attempt.current.message_id } })
    if (!selectedId) { appStorage.setItem('camelid.codingSession', next.id); selected.current = next.id; setSelectedId(next.id) }
    accept(next); attempt.current = null; return next
  })
  const control = action => mutate(async () => accept(await codingRequest(apiBase, '/' + encodeURIComponent(selectedId) + '/control', { method: 'POST', body: { action } })))
  const decide = approved => mutate(async () => {
    const id = snapshot?.approval?.id
    if (!id || decidingId === id) return
    setDecidingId(id)
    try { await codingRequest(apiBase, '/' + encodeURIComponent(selectedId) + '/approvals/' + encodeURIComponent(id), { method: 'POST', body: { approved } }) }
    catch (e) { setDecidingId(''); throw e }
  })
  const remove = () => mutate(async () => {
    await codingRequest(apiBase, '/' + encodeURIComponent(selectedId), { method: 'DELETE' })
    select(''); await refresh()
  })
  const retry = () => mutate(async () => {
    await refresh()
    if (selectedId) { accept(await codingRequest(apiBase, '/' + encodeURIComponent(selectedId))); setConnection('connected') }
  })
  return { snapshot, sessions, toolCapableModel, selectedId, select, send, control, decide, remove, retry, busy, error, connection, decidingId }
}
