import assert from 'node:assert/strict'

class MemoryStorage {
  constructor(seed = {}) {
    this.values = new Map(Object.entries(seed))
  }

  get length() { return this.values.size }
  key(index) { return [...this.values.keys()][index] ?? null }
  getItem(key) { return this.values.has(String(key)) ? this.values.get(String(key)) : null }
  setItem(key, value) { this.values.set(String(key), String(value)) }
  removeItem(key) { this.values.delete(String(key)) }
  clear() { this.values.clear() }
}

class FullStorage extends MemoryStorage {
  setItem() { throw new Error('quota exceeded') }
}

const nativeDocument = { initialized: false, values: {} }
const invoke = async (command, args = {}) => {
  if (command === 'read_ui_storage') {
    return { version: 1, initialized: nativeDocument.initialized, values: { ...nativeDocument.values } }
  }
  if (command === 'replace_ui_storage') {
    nativeDocument.initialized = true
    nativeDocument.values = { ...args.values }
    return
  }
  if (command === 'set_ui_storage_value') {
    nativeDocument.initialized = true
    if (args.value === null) delete nativeDocument.values[args.key]
    else nativeDocument.values[args.key] = args.value
    return
  }
  throw new Error(`unexpected command: ${command}`)
}

function installWindow(storage, withDesktop = true) {
  globalThis.window = {
    localStorage: storage,
    ...(withDesktop ? { __TAURI__: { core: { invoke } } } : {}),
  }
}

// Launch one is served from one ephemeral port and starts with that origin's
// empty browser storage.
installWindow(new MemoryStorage())
const firstLaunch = await import('../src/lib/appStorage.js?desktop-launch=one')
await firstLaunch.hydrateAppStorage()
firstLaunch.appStorage.setItem('camelid-theme', 'light')
firstLaunch.appStorage.setItem('camelid.conversations', '[{"id":"chat-653"}]')
await firstLaunch.flushAppStorage()

// Launch two models a different port: its localStorage is unrelated and empty,
// but hydration must restore the native document before app state initializes.
const secondOrigin = new MemoryStorage()
installWindow(secondOrigin)
const secondLaunch = await import('../src/lib/appStorage.js?desktop-launch=two')
await secondLaunch.hydrateAppStorage()
assert.equal(secondLaunch.appStorage.getItem('camelid-theme'), 'light')
assert.equal(secondLaunch.appStorage.getItem('camelid.conversations'), '[{"id":"chat-653"}]')

// Native state remains readable when the WebView origin cache is full and
// cannot accept the hydrated conversation payload.
installWindow(new FullStorage())
const fullOriginLaunch = await import('../src/lib/appStorage.js?desktop-launch=full-origin')
await fullOriginLaunch.hydrateAppStorage()
assert.equal(fullOriginLaunch.appStorage.getItem('camelid-theme'), 'light')
assert.equal(fullOriginLaunch.appStorage.getItem('camelid.conversations'), '[{"id":"chat-653"}]')

// Once initialized, an empty native document is authoritative. A coincidentally
// reused port must not resurrect values deleted during a previous launch.
secondLaunch.appStorage.clear()
await secondLaunch.flushAppStorage()
const staleOrigin = new MemoryStorage({ 'camelid-theme': 'dark', 'camelid.conversations': '[{"id":"stale"}]' })
installWindow(staleOrigin)
const thirdLaunch = await import('../src/lib/appStorage.js?desktop-launch=three')
await thirdLaunch.hydrateAppStorage()
assert.equal(thirdLaunch.appStorage.getItem('camelid-theme'), null)
assert.equal(thirdLaunch.appStorage.getItem('camelid.conversations'), null)

// The normal browser build remains a direct localStorage fallback.
const browserOrigin = new MemoryStorage()
installWindow(browserOrigin, false)
const browserLaunch = await import('../src/lib/appStorage.js?browser-launch')
await browserLaunch.hydrateAppStorage()
browserLaunch.appStorage.setItem('camelid-theme', 'system')
assert.equal(browserOrigin.getItem('camelid-theme'), 'system')

console.log('desktop storage smoke: PASS')
