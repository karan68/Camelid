/* React binding for the InferenceTelemetryStore: one store instance per
   mount, connected to the active API base. Lifecycle events re-render
   immediately via the store's subscription; fast-changing live metrics are
   sampled on a coarse interval so token-rate events never thrash React. */

import { useEffect, useMemo, useReducer } from 'react'
import { createInferenceTelemetryStore } from '../lib/inferenceTelemetry'

const LIVE_SAMPLE_MS = 250

export function useInferenceTelemetry(apiBase) {
  const store = useMemo(() => createInferenceTelemetryStore(), [])
  const [, bump] = useReducer((n) => n + 1, 0)

  useEffect(() => {
    store.connect(apiBase)
    const unsubscribe = store.subscribe(bump)
    return () => {
      unsubscribe()
      store.disconnect()
    }
  }, [store, apiBase])

  useEffect(() => {
    const interval = window.setInterval(() => {
      const run = store.getRun()
      if (run.active || store.getConnection() !== 'live') bump()
    }, LIVE_SAMPLE_MS)
    return () => window.clearInterval(interval)
  }, [store])

  return store
}
