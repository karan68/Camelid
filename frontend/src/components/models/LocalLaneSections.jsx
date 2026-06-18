import { useCallback, useEffect, useState } from 'react'
import { isCompatibilitySupportedForModel } from '../../lib/capabilities'
import { EvidenceChip } from '../ui/EvidenceChip'
import { ParityReceiptCard } from '../chat/render/ParityReceipt'
import { UnsupportedBlocker } from './UnsupportedBlocker'

/* Local models on disk, grouped into derived lane sections. Membership is computed
   from /api/models/local lane facts + the /api/capabilities contract — never a
   hand-authored array. Copper is reserved for supported; runnable is amber and never
   copper; the not-yet-runnable state is shown, never hidden. */

const GB = 1024 * 1024 * 1024

function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

/* A model object shaped for the existing contract matcher (it reads id/name/
   model_path/quant). The supported gate stays the contract's voice — we only ask it. */
function matchModel(entry) {
  return {
    id: entry.filename,
    name: entry.filename,
    model_path: entry.filename,
    hf_filename: entry.filename,
    quant: entry.quantization,
  }
}

function laneOf(entry, capabilities) {
  if (isCompatibilitySupportedForModel(capabilities, matchModel(entry))) return 'supported'
  if (entry.runnable_receipt_present) return 'compatible'
  if (entry.admitted && entry.oracle_qualified) return 'eligible'
  return 'not_anchored'
}

function metaLine(entry) {
  const ctx = entry.context_length
    ? `${entry.context_length >= 1000 ? `${Math.round(entry.context_length / 1000)}K` : entry.context_length} ctx`
    : null
  return [entry.architecture, entry.quantization, entry.tokenizer_kind, prettySize(entry.size_bytes), ctx]
    .filter(Boolean)
    .join(' · ')
}

/* What the MODEL is GOOD AT — its strengths/use-cases, by family. Independent of any
   system, hardware, or lane: this describes the model, not where it runs. */
function describeModel(entry) {
  const name = (entry.filename || '').toLowerCase()
  if (name.includes('mistral')) return 'Good at reasoning, coding, and following detailed instructions.'
  if (name.includes('tinyllama')) return 'A tiny model for quick, simple chat and experiments.'
  switch (entry.architecture) {
    case 'qwen3':
      return 'Good at multilingual chat, reasoning, coding, and math.'
    case 'gemma':
    case 'gemma3':
    case 'gemma4':
      return 'Good at multilingual chat and general reasoning.'
    case 'phi3':
      return 'Good at reasoning, math, and coding in a compact model.'
    case 'llama':
      return 'Good at everyday chat, summarizing, and writing.'
    default:
      return entry.chat_capable
        ? 'Good at general chat and instruction following.'
        : 'Text generation.'
  }
}

function Section({ title, subtitle, count, children }) {
  return (
    <section className="lane-section">
      <header className="lane-section-head">
        <h3>
          {title} <span className="lane-section-count">{count}</span>
        </h3>
        <p className="lane-section-sub">{subtitle}</p>
      </header>
      <div className="lane-section-body">{children}</div>
    </section>
  )
}

function SupportedRow({ entry, active, busy, onUse }) {
  return (
    <article
      className={`lane-row lane-row--supported${active ? ' lane-row--active' : ''}`}
      aria-label={`Supported model ${entry.filename}`}
    >
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="supported" asText>Supported</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      {active ? (
        <p className="lane-row-loaded">● Loaded — this is the active chat model.</p>
      ) : (
        <button type="button" className="lane-row-action" onClick={onUse} disabled={busy}>
          {busy ? 'Loading…' : 'Use for chat'}
        </button>
      )}
    </article>
  )
}

function CompatibleRow({ entry, receipt }) {
  return (
    <article className="lane-row lane-row--runnable" aria-label={`Compatible model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>Runnable</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      {receipt ? (
        <ParityReceiptCard receipt={receipt} />
      ) : (
        <p className="lane-row-faint">Loading runnable receipt…</p>
      )}
      <p className="lane-row-faint">
        Runnable execution is the generic f32 lane — run it with the CLI
        (<code>camelid runnable-smoke</code>). No HTTP serve endpoint yet, so there is no
        in-app “Use for chat” for the runnable lane.
      </p>
    </article>
  )
}

function EligibleRow({ entry, busy, onRun }) {
  return (
    <article className="lane-row lane-row--runnable" aria-label={`Smoke-eligible model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>Oracle-qualified</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      <button type="button" className="lane-row-action" onClick={onRun} disabled={busy}>
        {busy ? 'Running smoke-admission…' : 'Run smoke-admission'}
      </button>
    </article>
  )
}

function NotAnchoredRow({ entry }) {
  return (
    <article className="lane-row lane-row--blocked" aria-label={`Not-yet-runnable model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="unsupported" asText>Combo not yet anchored</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
    </article>
  )
}

export function LocalLaneSections({ apiBase = '', capabilities, refreshKey = 0 }) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [receipts, setReceipts] = useState({})
  const [busy, setBusy] = useState({})
  const [activeFilename, setActiveFilename] = useState('')
  const [usingFilename, setUsingFilename] = useState('')
  // Typed fail-closed blocker from a pre-load inspect ({ code, message }), shown
  // verbatim instead of attempting a multi-GB load that cannot run.
  const [blocker, setBlocker] = useState(null)

  const refreshCurrent = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models/current`)
      if (!res.ok) return
      const cur = await res.json()
      const path = String(cur?.path || '')
      setActiveFilename(path.split(/[\\/]/).pop() || '')
    } catch {
      /* best-effort — no active highlight if unavailable */
    }
  }, [base])

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${base}/api/models/local`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setLoading(false)
    }
  }, [base])

  // Load a local model into the chat backend. First predict the lane with a
  // header-only inspect (no multi-GB read): if the architecture is not implemented,
  // surface the exact typed blocker and stop — never attempt to run it. Implemented
  // architectures (supported or experimental) load as before.
  const useModel = async (filename) => {
    setUsingFilename(filename)
    setError('')
    setBlocker(null)
    const path = `${data?.models_dir || 'models'}/${filename}`
    try {
      const inspectRes = await fetch(`${base}/api/models/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      if (inspectRes.ok) {
        const inspect = await inspectRes.json()
        if (inspect?.blocker) {
          setBlocker(inspect.blocker)
          return
        }
      }
      // Implemented (or inspect unavailable) → attempt the real load.
      const res = await fetch(`${base}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: filename, path }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // A typed fail-closed load error (e.g. invalid metadata) becomes a blocker.
        if (body?.error?.code && body.error.code !== 'invalid_model') {
          setBlocker({ code: body.error.code, message: body.error.message })
          return
        }
        throw new Error(body?.error?.message || `load failed (HTTP ${res.status})`)
      }
      await refreshCurrent()
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setUsingFilename('')
    }
  }

  useEffect(() => {
    refresh()
    refreshCurrent()
  }, [refresh, refreshCurrent, refreshKey])

  // Pull the runnable receipt for each Compatible model (those that passed smoke).
  useEffect(() => {
    if (!data) return
    data.models
      .filter((m) => m.runnable_receipt_present && !receipts[m.filename])
      .forEach(async (m) => {
        try {
          const res = await fetch(
            `${base}/api/models/runnable-receipt?filename=${encodeURIComponent(m.filename)}`,
          )
          if (res.ok) {
            const receipt = await res.json()
            setReceipts((r) => ({ ...r, [m.filename]: receipt }))
          }
        } catch {
          /* receipt is best-effort; the row still renders */
        }
      })
  }, [data, base, receipts])

  const runSmoke = async (filename) => {
    setBusy((b) => ({ ...b, [filename]: true }))
    setError('')
    try {
      const res = await fetch(`${base}/api/models/runnable-smoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
      const body = await res.json()
      if (res.ok && body.passed) {
        setReceipts((r) => ({ ...r, [filename]: body.receipt }))
        await refresh()
      } else {
        setError(body?.error?.message || `Smoke-admission did not pass for ${filename}.`)
      }
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setBusy((b) => ({ ...b, [filename]: false }))
    }
  }

  if (loading && !data) {
    return <p className="lane-empty">Scanning local models…</p>
  }
  if (error && !data) {
    return <p className="lane-empty">Could not list local models: {error}</p>
  }
  if (!data) return null

  const buckets = { supported: [], compatible: [], eligible: [], not_anchored: [] }
  for (const m of data.models) buckets[laneOf(m, capabilities)].push(m)

  return (
    <div className="local-lane-sections">
      <div className="local-lane-head">
        <h2>Local models by lane</h2>
        <button type="button" className="lane-refresh" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      <p className="local-lane-intro">
        Derived from <code>/api/models/local</code> + the support contract — membership is computed
        from receipts and lane status, never hand-authored.
      </p>
      {error ? <p className="lane-error">{error}</p> : null}
      {blocker ? (
        <UnsupportedBlocker blocker={blocker} className="local-lane-blocker" />
      ) : null}

      <Section
        title="Supported"
        count={buckets.supported.length}
        subtitle="Cross-validated supported-lane parity. Copper."
      >
        {buckets.supported.length ? (
          buckets.supported.map((m) => (
            <SupportedRow
              key={m.filename}
              entry={m}
              active={m.filename === activeFilename}
              busy={usingFilename === m.filename}
              onUse={() => useModel(m.filename)}
            />
          ))
        ) : (
          <p className="lane-empty">No on-disk model matches a supported parity row.</p>
        )}
      </Section>

      <Section
        title="Compatible"
        count={buckets.compatible.length}
        subtitle="Passed smoke-admission on an oracle-qualified combo. Runnable lane — deterministic execution attested, NOT parity. Never copper."
      >
        {buckets.compatible.length ? (
          buckets.compatible.map((m) => (
            <CompatibleRow key={m.filename} entry={m} receipt={receipts[m.filename]} />
          ))
        ) : (
          <p className="lane-empty">No model has passed smoke-admission yet.</p>
        )}
      </Section>

      {buckets.eligible.length ? (
        <Section
          title="Run smoke-admission"
          count={buckets.eligible.length}
          subtitle="Oracle-qualified combos not yet smoked. Run the check to admit them to Compatible."
        >
          {buckets.eligible.map((m) => (
            <EligibleRow
              key={m.filename}
              entry={m}
              busy={Boolean(busy[m.filename])}
              onRun={() => runSmoke(m.filename)}
            />
          ))}
        </Section>
      ) : null}

      {buckets.not_anchored.length ? (
        <Section
          title="Not yet runnable"
          count={buckets.not_anchored.length}
          subtitle="Combo not anchored — shown for honesty, never presented as compatible."
        >
          {buckets.not_anchored.map((m) => (
            <NotAnchoredRow key={m.filename} entry={m} />
          ))}
        </Section>
      ) : null}
    </div>
  )
}
