import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
/* Self-hosted fonts (Fontsource) — bundled locally; the app must render fully
   offline, so no third-party font CDN imports anywhere. */
import '@fontsource-variable/inter'
import '@fontsource-variable/space-grotesk'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@fontsource/ibm-plex-mono/600.css'
import './styles.css'
import { installApiAuthFetch } from './lib/apiAuth'
import { hydrateAppStorage } from './lib/appStorage.js'

async function startApp() {
  // Camelid Desktop serves this UI from a new loopback port after each restart.
  // Restore the app-owned storage document before any hook reads origin-scoped
  // localStorage; the regular browser build resolves this immediately.
  await hydrateAppStorage()
  installApiAuthFetch()

  ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

startApp()
