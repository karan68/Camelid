const CAMELID_STORAGE_PREFIX = 'camelid'

let desktopInvoke = null
let desktopValues = null
let hydrated = false
let persistenceTail = Promise.resolve()
let persistenceWarningShown = false

function browserStorage() {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

function isCamelidKey(key) {
  return String(key).startsWith(CAMELID_STORAGE_PREFIX)
}

function readCamelidEntries(storage) {
  const entries = {}
  if (!storage) return entries
  try {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index)
      if (!key || !isCamelidKey(key)) continue
      const value = storage.getItem(key)
      if (value !== null) entries[key] = value
    }
  } catch {
    // Browser storage remains best-effort in privacy-restricted contexts.
  }
  return entries
}

function replaceBrowserCamelidEntries(storage, values) {
  if (!storage) return
  try {
    for (const key of Object.keys(readCamelidEntries(storage))) storage.removeItem(key)
    for (const [key, value] of Object.entries(values)) storage.setItem(key, String(value))
  } catch {
    // The native document is still authoritative in Desktop even if the
    // current WebView origin refuses its local compatibility cache.
  }
}

function warnPersistenceFailure(error) {
  if (persistenceWarningShown || typeof console === 'undefined') return
  persistenceWarningShown = true
  console.warn('Camelid Desktop could not persist UI settings.', error)
}

function queueDesktopWrite(command, args) {
  if (!desktopInvoke) return
  persistenceTail = persistenceTail
    .then(() => desktopInvoke(command, args))
    .catch((error) => warnPersistenceFailure(error))
}

/**
 * Hydrate the current ephemeral WebView origin before React reads any settings.
 * The native app-data document is canonical after its first initialization;
 * the first launch alone imports any existing Camelid localStorage values.
 */
export async function hydrateAppStorage() {
  if (hydrated || typeof window === 'undefined') return
  hydrated = true
  const invoke = window.__TAURI__?.core?.invoke
  if (!invoke) return

  const storage = browserStorage()
  try {
    const snapshot = await invoke('read_ui_storage')
    desktopInvoke = invoke
    const persisted = snapshot?.values && typeof snapshot.values === 'object'
      ? snapshot.values
      : {}

    if (snapshot?.initialized) {
      desktopValues = new Map(Object.entries(persisted).map(([key, value]) => [key, String(value)]))
      replaceBrowserCamelidEntries(storage, persisted)
    } else {
      const imported = readCamelidEntries(storage)
      await invoke('replace_ui_storage', { values: imported })
      desktopValues = new Map(Object.entries(imported))
    }
  } catch (error) {
    desktopInvoke = null
    desktopValues = null
    warnPersistenceFailure(error)
  }
}

export const appStorage = {
  getItem(key) {
    const normalizedKey = String(key)
    if (desktopValues && isCamelidKey(normalizedKey)) {
      return desktopValues.get(normalizedKey) ?? null
    }
    try {
      return browserStorage()?.getItem(normalizedKey) ?? null
    } catch {
      return null
    }
  },

  setItem(key, value) {
    const normalizedKey = String(key)
    const normalizedValue = String(value)
    if (desktopValues && isCamelidKey(normalizedKey)) {
      desktopValues.set(normalizedKey, normalizedValue)
    }
    try {
      browserStorage()?.setItem(normalizedKey, normalizedValue)
    } catch {
      // Preserve the existing best-effort browser behavior.
    }
    if (isCamelidKey(normalizedKey)) {
      queueDesktopWrite('set_ui_storage_value', { key: normalizedKey, value: normalizedValue })
    }
  },

  removeItem(key) {
    const normalizedKey = String(key)
    if (desktopValues && isCamelidKey(normalizedKey)) {
      desktopValues.delete(normalizedKey)
    }
    try {
      browserStorage()?.removeItem(normalizedKey)
    } catch {
      // Preserve the existing best-effort browser behavior.
    }
    if (isCamelidKey(normalizedKey)) {
      queueDesktopWrite('set_ui_storage_value', { key: normalizedKey, value: null })
    }
  },

  clear() {
    desktopValues?.clear()
    try {
      browserStorage()?.clear()
    } catch {
      // Preserve the existing best-effort browser behavior.
    }
    queueDesktopWrite('replace_ui_storage', { values: {} })
  },
}

// Exposed for deterministic smoke coverage and callers that need a durability barrier.
export function flushAppStorage() {
  return persistenceTail
}
