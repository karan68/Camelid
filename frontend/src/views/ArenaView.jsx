import { useState, useRef, useEffect } from 'react'
import { IconSend, IconStop, IconScale, IconBolt, IconRefresh, IconCheck } from '../components/ui/icons'
import { AssistantMarkdown } from '../lib/markdown'
import { formatModelLabel } from '../lib/formatters'

const PRESET_PROMPTS = [
  'Explain how speculative decoding works in local LLMs in 3 concise bullet points.',
  'Write a clean, memory-safe LRU cache implementation in Rust with unit tests.',
  'Compare the architectural trade-offs of Mixtral MoE vs dense Llama 3.2.',
  'Draft a persuasive, professional email proposing local offline AI inference for our engineering team.',
]

export default function ArenaView({ models = [], runtime, capabilities }) {
  const [modelA, setModelA] = useState(() => models[0]?.id || '')
  const [modelB, setModelB] = useState(() => models[1]?.id || models[0]?.id || '')
  const [prompt, setPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)

  const [stateA, setStateA] = useState({
    output: '',
    ttft: null,
    tokensPerSec: null,
    totalTokens: 0,
    elapsedMs: 0,
    status: 'idle', // 'idle' | 'generating' | 'done' | 'error'
    error: '',
  })

  const [stateB, setStateB] = useState({
    output: '',
    ttft: null,
    tokensPerSec: null,
    totalTokens: 0,
    elapsedMs: 0,
    status: 'idle',
    error: '',
  })

  const [vote, setVote] = useState(null)
  const abortControllersRef = useRef({ a: null, b: null })

  // Keep defaults updated if models list loads later
  useEffect(() => {
    if (!modelA && models.length > 0) setModelA(models[0].id)
    if (!modelB && models.length > 1) setModelB(models[1].id)
    else if (!modelB && models.length > 0) setModelB(models[0].id)
  }, [models, modelA, modelB])

  const stopAll = () => {
    if (abortControllersRef.current.a) {
      abortControllersRef.current.a.abort()
      abortControllersRef.current.a = null
    }
    if (abortControllersRef.current.b) {
      abortControllersRef.current.b.abort()
      abortControllersRef.current.b = null
    }
    setIsGenerating(false)
    setStateA((s) => ({ ...s, status: s.status === 'generating' ? 'done' : s.status }))
    setStateB((s) => ({ ...s, status: s.status === 'generating' ? 'done' : s.status }))
  }

  const streamModel = async (modelId, isA) => {
    const setState = isA ? setStateA : setStateB
    const controller = new AbortController()
    if (isA) abortControllersRef.current.a = controller
    else abortControllersRef.current.b = controller

    const startTime = performance.now()
    let firstTokenTime = null
    let count = 0

    setState({
      output: '',
      ttft: null,
      tokensPerSec: null,
      totalTokens: 0,
      elapsedMs: 0,
      status: 'generating',
      error: '',
    })

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          temperature: 0.7,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let partial = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        partial += chunk
        const lines = partial.split('\n')
        partial = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data:')) continue
          const dataStr = trimmed.slice(5).trim()
          if (dataStr === '[DONE]') break

          try {
            const parsed = JSON.parse(dataStr)
            const token = parsed.choices?.[0]?.delta?.content || ''
            if (token) {
              if (firstTokenTime === null) {
                firstTokenTime = performance.now()
                const ttft = Math.round(firstTokenTime - startTime)
                setState((s) => ({ ...s, ttft }))
              }
              count += 1
              const now = performance.now()
              const elapsedSeconds = Math.max(0.001, (now - firstTokenTime) / 1000)
              const tps = count > 1 ? (count / elapsedSeconds).toFixed(1) : null

              setState((s) => ({
                ...s,
                output: s.output + token,
                totalTokens: count,
                tokensPerSec: tps,
                elapsedMs: Math.round(now - startTime),
              }))
            }
          } catch {
            // Partial JSON ignored
          }
        }
      }

      setState((s) => ({ ...s, status: 'done' }))
    } catch (err) {
      if (err.name === 'AbortError') {
        setState((s) => ({ ...s, status: 'done' }))
      } else {
        setState((s) => ({ ...s, status: 'error', error: err.message || 'Generation failed' }))
      }
    }
  }

  const handleSend = async (e) => {
    e?.preventDefault()
    if (!prompt.trim() || isGenerating) return
    setVote(null)
    setIsGenerating(true)

    await Promise.allSettled([
      streamModel(modelA, true),
      streamModel(modelB, false),
    ])

    setIsGenerating(false)
  }

  return (
    <div className="arena-view" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '20px', gap: '16px', overflow: 'hidden' }}>
      <header className="arena-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flex: 'none' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'var(--color-accent, #38bdf8)', color: '#000', padding: '6px', borderRadius: '8px', display: 'grid', placeItems: 'center' }}>
            <IconScale size={20} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--color-text)' }}>Model Arena</h1>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--color-text-muted)' }}>Side-by-side split comparison with live TTFT and tokens/sec telemetry</p>
          </div>
        </div>

        {vote && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 12px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid var(--color-accent, #38bdf8)', borderRadius: '20px', color: 'var(--color-accent, #38bdf8)', fontSize: '12px', fontWeight: 600 }}>
            <IconCheck size={14} /> Voted: {vote}
          </div>
        )}
      </header>

      {/* Arena Split Panes */}
      <div className="arena-panes" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', flex: 1, minHeight: 0 }}>
        {/* Model A Pane */}
        <div className="arena-card" style={{ display: 'flex', flexDirection: 'column', background: 'var(--color-surface, #1e242c)', border: '1px solid var(--color-border-soft, #2a333f)', borderRadius: '12px', overflow: 'hidden' }}>
          <div className="arena-card__header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'rgba(0, 0, 0, 0.15)', borderBottom: '1px solid var(--color-border-soft, #2a333f)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 8px', borderRadius: '4px', background: '#38bdf8', color: '#000' }}>Model A</span>
              <select
                value={modelA}
                onChange={(e) => setModelA(e.target.value)}
                disabled={isGenerating}
                style={{ background: 'var(--color-surface-sunken, #151a21)', color: 'var(--color-text)', border: '1px solid var(--color-border-soft, #2a333f)', borderRadius: '6px', padding: '4px 8px', fontSize: '13px' }}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{formatModelLabel(m.name || m.id)}</option>
                ))}
              </select>
            </div>
            {/* Telemetry Chips */}
            <div style={{ display: 'flex', gap: '8px', fontSize: '11px', color: 'var(--color-text-faint)' }}>
              {stateA.ttft !== null && (
                <span title="Time To First Token" style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  TTFT: <strong style={{ color: 'var(--color-accent, #38bdf8)' }}>{stateA.ttft}ms</strong>
                </span>
              )}
              {stateA.tokensPerSec && (
                <span title="Generation Speed" style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  Speed: <strong style={{ color: '#4ade80' }}>{stateA.tokensPerSec} tok/s</strong>
                </span>
              )}
              {stateA.totalTokens > 0 && (
                <span style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  {stateA.totalTokens} toks
                </span>
              )}
            </div>
          </div>
          <div className="arena-card__body" style={{ flex: 1, padding: '16px', overflowY: 'auto', fontSize: '14px', lineHeight: 1.6, color: 'var(--color-text)' }}>
            {stateA.output ? (
              <AssistantMarkdown content={stateA.output} />
            ) : stateA.status === 'generating' ? (
              <div style={{ color: 'var(--color-text-faint)', fontStyle: 'italic' }}>Generating response...</div>
            ) : stateA.error ? (
              <div style={{ color: '#f87171' }}>Error: {stateA.error}</div>
            ) : (
              <div style={{ color: 'var(--color-text-faint)', textAlign: 'center', marginTop: '40px' }}>Responses for Model A will appear here.</div>
            )}
          </div>
        </div>

        {/* Model B Pane */}
        <div className="arena-card" style={{ display: 'flex', flexDirection: 'column', background: 'var(--color-surface, #1e242c)', border: '1px solid var(--color-border-soft, #2a333f)', borderRadius: '12px', overflow: 'hidden' }}>
          <div className="arena-card__header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'rgba(0, 0, 0, 0.15)', borderBottom: '1px solid var(--color-border-soft, #2a333f)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 8px', borderRadius: '4px', background: '#a855f7', color: '#fff' }}>Model B</span>
              <select
                value={modelB}
                onChange={(e) => setModelB(e.target.value)}
                disabled={isGenerating}
                style={{ background: 'var(--color-surface-sunken, #151a21)', color: 'var(--color-text)', border: '1px solid var(--color-border-soft, #2a333f)', borderRadius: '6px', padding: '4px 8px', fontSize: '13px' }}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{formatModelLabel(m.name || m.id)}</option>
                ))}
              </select>
            </div>
            {/* Telemetry Chips */}
            <div style={{ display: 'flex', gap: '8px', fontSize: '11px', color: 'var(--color-text-faint)' }}>
              {stateB.ttft !== null && (
                <span title="Time To First Token" style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  TTFT: <strong style={{ color: 'var(--color-accent, #38bdf8)' }}>{stateB.ttft}ms</strong>
                </span>
              )}
              {stateB.tokensPerSec && (
                <span title="Generation Speed" style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  Speed: <strong style={{ color: '#4ade80' }}>{stateB.tokensPerSec} tok/s</strong>
                </span>
              )}
              {stateB.totalTokens > 0 && (
                <span style={{ background: 'rgba(255, 255, 255, 0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                  {stateB.totalTokens} toks
                </span>
              )}
            </div>
          </div>
          <div className="arena-card__body" style={{ flex: 1, padding: '16px', overflowY: 'auto', fontSize: '14px', lineHeight: 1.6, color: 'var(--color-text)' }}>
            {stateB.output ? (
              <AssistantMarkdown content={stateB.output} />
            ) : stateB.status === 'generating' ? (
              <div style={{ color: 'var(--color-text-faint)', fontStyle: 'italic' }}>Generating response...</div>
            ) : stateB.error ? (
              <div style={{ color: '#f87171' }}>Error: {stateB.error}</div>
            ) : (
              <div style={{ color: 'var(--color-text-faint)', textAlign: 'center', marginTop: '40px' }}>Responses for Model B will appear here.</div>
            )}
          </div>
        </div>
      </div>

      {/* Post-battle Voting Bar */}
      {!isGenerating && stateA.output && stateB.output && !vote && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', padding: '8px', background: 'rgba(0, 0, 0, 0.2)', borderRadius: '8px', border: '1px solid var(--color-border-soft)' }}>
          <span style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>Which model gave a better answer?</span>
          <button type="button" className="cxcomposer__tool" onClick={() => setVote('Model A')}>👈 Model A was better</button>
          <button type="button" className="cxcomposer__tool" onClick={() => setVote('Tie')}>🤝 Both Equal / Tie</button>
          <button type="button" className="cxcomposer__tool" onClick={() => setVote('Model B')}>👉 Model B was better</button>
        </div>
      )}

      {/* Bottom Composer & Presets */}
      <footer className="arena-footer" style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 'none' }}>
        {/* Preset Prompts Pills */}
        <div style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '4px' }}>
          {PRESET_PROMPTS.map((p, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setPrompt(p)}
              disabled={isGenerating}
              style={{ flex: 'none', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid var(--color-border-soft)', borderRadius: '16px', padding: '4px 10px', fontSize: '11px', color: 'var(--color-text-muted)', cursor: 'pointer', whiteSpace: 'nowrap' }}
            >
              <IconBolt size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {p.length > 40 ? p.slice(0, 40) + '...' : p}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <form onSubmit={handleSend} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={isGenerating}
            placeholder="Type a battle prompt to send to both models simultaneously..."
            style={{ flex: 1, padding: '12px 16px', background: 'var(--color-surface, #1e242c)', border: '1px solid var(--color-border-soft, #2a333f)', borderRadius: '10px', color: 'var(--color-text)', fontSize: '14px' }}
          />
          {isGenerating ? (
            <button
              type="button"
              onClick={stopAll}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '12px 20px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: '10px', fontWeight: 600, cursor: 'pointer' }}
            >
              <IconStop size={16} /> Stop Both
            </button>
          ) : (
            <button
              type="submit"
              disabled={!prompt.trim()}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '12px 20px', background: 'var(--color-accent, #38bdf8)', color: '#000', border: 'none', borderRadius: '10px', fontWeight: 600, cursor: prompt.trim() ? 'pointer' : 'not-allowed', opacity: prompt.trim() ? 1 : 0.5 }}
            >
              <IconSend size={16} /> Compare
            </button>
          )}
        </form>
      </footer>
    </div>
  )
}
