/* Shared runtime-status vocabulary. Every surface that names the engine state
   (System tiles, API tiles, headers) must agree on one word per state — the
   same runtime must never read "Idle" on one page and "Offline" on another. */

export const RUNTIME_STATUS = {
  ready: { label: 'Ready', tone: 'ready' },
  loaded: { label: 'Loaded', tone: 'warn' },
  idle: { label: 'Idle', tone: 'neutral' },
  offline: { label: 'Offline', tone: 'offline' },
}

export function runtimeStatusKey(runtime) {
  if (runtime?.generation_ready) return 'ready'
  if (runtime?.loaded_now) return 'loaded'
  if (runtime?.status === 'offline') return 'offline'
  return 'idle'
}

export function describeRuntimeStatus(runtime) {
  return RUNTIME_STATUS[runtimeStatusKey(runtime)]
}
