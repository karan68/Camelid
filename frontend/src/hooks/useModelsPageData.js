import { useCallback, useEffect, useRef, useState } from 'react'

/* Single data spine for the Models page. Owns fetch + refresh for the three
   backend truths the page renders from — /api/models/local (disk scan with lane
   facts), /api/models/current (loaded model), /api/models/catalog/downloads
   (live download progress) — so every zone reads the same snapshot instead of
   fetching privately. Capabilities stay a prop (already lifted in
   useDashboardData); this hook never touches localStorage: "downloaded" is only
   ever the live disk scan.

   The downloads poll runs only while there is (or was just started) an active
   download and backs off when idle. When a download leaves the backend's active
   list (terminal statuses are reported once, then dropped), the local scan is
   refreshed so the file appears in its derived lane without a page reload. */

const DOWNLOAD_POLL_MS = 1500
// Ticks the poller keeps running after the list goes empty, so a just-started
// install (kicked before the backend registers it) isn't missed.
const IDLE_TICKS_BEFORE_STOP = 3

export function useModelsPageData({ apiBase = '' } = {}) {
  const base = (apiBase || '').replace(/\/$/, '')

  const [local, setLocal] = useState(null) // { models_dir, models: [...] }
  const [localLoading, setLocalLoading] = useState(false)
  const [localError, setLocalError] = useState('')
  const [current, setCurrent] = useState(null) // { path } from /api/models/current
  const [downloads, setDownloads] = useState([])
  const [polling, setPolling] = useState(true) // one initial pass on mount

  const idleTicksRef = useRef(0)
  const prevDownloadIdsRef = useRef(new Set())

  const refreshLocal = useCallback(async () => {
    setLocalLoading(true)
    setLocalError('')
    try {
      const res = await fetch(`${base}/api/models/local`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setLocal(await res.json())
    } catch (err) {
      setLocalError(String(err?.message || err))
    } finally {
      setLocalLoading(false)
    }
  }, [base])

  const refreshCurrent = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/current`)
      if (!res.ok) {
        setCurrent(null)
        return
      }
      setCurrent(await res.json())
    } catch {
      setCurrent(null)
    }
  }, [base])

  const refreshDownloads = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/catalog/downloads`)
      if (!res.ok) return null
      const list = await res.json()
      setDownloads(Array.isArray(list) ? list : [])
      return Array.isArray(list) ? list : []
    } catch {
      return null
    }
  }, [base])

  /* Wake the downloads poller — call right after starting an install. */
  const kickDownloadsPoll = useCallback(() => {
    idleTicksRef.current = 0
    setPolling(true)
  }, [])

  const cancelDownload = useCallback(
    async (id) => {
      const res = await fetch(`${base}/api/models/catalog/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      await refreshDownloads()
      await refreshLocal()
      return res.ok
    },
    [base, refreshDownloads, refreshLocal],
  )

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshLocal(), refreshCurrent(), refreshDownloads()])
  }, [refreshLocal, refreshCurrent, refreshDownloads])

  useEffect(() => {
    refreshLocal()
    refreshCurrent()
  }, [refreshLocal, refreshCurrent])

  useEffect(() => {
    if (!polling) return undefined
    let cancelled = false
    const tick = async () => {
      const list = await refreshDownloads()
      if (cancelled || list === null) return
      const ids = new Set(list.filter((d) => d.status === 'downloading').map((d) => d.id))
      // Any download that left the active list (completed, failed, or canceled)
      // may have landed a file — re-scan disk so lanes update without a reload.
      const settled = [...prevDownloadIdsRef.current].some((id) => !ids.has(id))
      prevDownloadIdsRef.current = ids
      if (settled) {
        refreshLocal()
      }
      if (ids.size === 0) {
        idleTicksRef.current += 1
        if (idleTicksRef.current >= IDLE_TICKS_BEFORE_STOP) setPolling(false)
      } else {
        idleTicksRef.current = 0
      }
    }
    tick()
    const timer = setInterval(tick, DOWNLOAD_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [polling, refreshDownloads, refreshLocal])

  const activeFilename = String(current?.path || '').split(/[\\/]/).pop() || ''
  const localFilenames = new Set((local?.models || []).map((m) => m.filename))

  return {
    base,
    local,
    localLoading,
    localError,
    current,
    activeFilename,
    localFilenames,
    downloads,
    refreshLocal,
    refreshCurrent,
    refreshDownloads,
    refreshAll,
    kickDownloadsPoll,
    cancelDownload,
  }
}
