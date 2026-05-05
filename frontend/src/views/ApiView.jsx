import { capabilityStatusTone, compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, formatCapabilityStatus, getCurrentCompatibilityTarget, guardedCapabilityCopy, isGuardedCapabilityStatus, isSupportedCapabilityStatus, summarizeCapabilityItems } from '../lib/capabilities'

function guardedApiFeatures(features = []) {
  return features.filter((feature) => isGuardedCapabilityStatus(feature.status))
}

export default function ApiView({ runtime, selectedModel, capabilities }) {
  const apiBase = runtime?.api_base || ''
  const modelId = selectedModel?.id || runtime?.active_model_id || '<loaded-model-id>'
  const supportContract = capabilities?.support_contract
  const compatibilityTargets = capabilities?.model_compatibility || []
  const currentTarget = getCurrentCompatibilityTarget(capabilities)
  const apiFeatures = capabilities?.api_features || []
  const guardedFeatures = guardedApiFeatures(apiFeatures)
  const selectedCompatibilityHint = findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = selectedCompatibilityHint?.kind === 'compatibility' ? selectedCompatibilityHint.target : null
  const selectedCompatibilitySupported = selectedCompatibilityTarget ? isSupportedCapabilityStatus(selectedCompatibilityTarget.status) : false
  const generationReady = Boolean(runtime?.generation_ready)
  const loadedNow = Boolean(runtime?.loaded_now)

  return (
    <section className="view-stack view-shell api-view-shell">
      <div className="panel panel-hero system-hero system-hero-separated">
        <div className="view-hero-copy">
          <p className="panel-kicker">API</p>
          <h2>Local API contract and readiness</h2>
          <p className="hero-summary">This view makes the backend support contract explicit: /api/capabilities describes what Camelid has evidence for, while /v1/health decides whether the currently loaded model can actually chat.</p>
        </div>
        <div className="view-hero-stats system-hero-pills system-hero-pills-polished">
          <div className={`status-pill ${generationReady ? 'ready' : 'warm'}`}>{generationReady ? 'generation_ready=true' : loadedNow ? 'loaded_now=true · chat blocked' : 'no generation-ready model'}</div>
          <div className="status-pill">{apiBase || 'Local API unavailable'}</div>
        </div>
      </div>

      <section className="panel api-panel panel-section">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Endpoints</p>
            <h2>OpenAI-compatible surface</h2>
            <p className="hero-summary">Generation endpoints stay useful only when runtime readiness is green and the selected local GGUF has an exact supported compatibility row. Capability rows explain supported and guarded lanes, but they never override loaded_now/generation_ready or active_model_id matching.</p>
          </div>
          <div className={`status-pill ${generationReady ? 'ready' : 'warm'}`}>{generationReady ? 'Local /v1 generation ready' : 'Load a generation-ready model'}</div>
        </div>

        <div className="api-grid api-grid-polished">
          <div className="api-card">
            <strong>Chat completions</strong>
            <code>{apiBase ? `${apiBase}/v1/chat/completions` : 'Unavailable until the local API is running'}</code>
            <p>{generationReady ? 'Runnable now only for the loaded GGUF when /api/capabilities also matches an exact supported row.' : 'Do not call for UX chat yet; Camelid must report loaded_now=true and generation_ready=true for an exact supported row first.'}</p>
          </div>
          <div className="api-card">
            <strong>Model listing</strong>
            <code>{apiBase ? `${apiBase}/v1/models` : 'Unavailable until the local API is running'}</code>
            <p>Lists the active runtime model. It is not a broad compatibility catalog.</p>
          </div>
          <div className="api-card">
            <strong>Health</strong>
            <code>{apiBase ? `${apiBase}/v1/health` : 'Unavailable until the local API is running'}</code>
            <p>Source of truth for active_model_id, loaded_now, and generation_ready.</p>
          </div>
          <div className="api-card">
            <strong>Capabilities</strong>
            <code>{apiBase ? `${apiBase}/api/capabilities` : 'Unavailable until the local API is running'}</code>
            <p>Support contract for model families, quants, compatibility rows, API feature support, and typed guardrails.</p>
          </div>
          <div className="api-card wide api-card-code">
            <strong>Readiness-gated curl</strong>
            <pre>{apiBase ? `# Use after /v1/health returns generation_ready=true\ncurl ${apiBase}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "${modelId}",\n    "messages": [{"role": "user", "content": "Hello from Camelid"}],\n    "max_tokens": 16,\n    "temperature": 0\n  }'` : 'Start the local runtime to see a ready-to-copy curl example.'}</pre>
          </div>
        </div>
      </section>

      <section className="panel api-panel panel-section">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Support contract</p>
            <h2>/api/capabilities summary</h2>
            <p className="hero-summary">The UI treats these rows as evidence boundaries, not marketing claims. Planned, partial, blocked, or unsupported rows remain visible but guarded.</p>
          </div>
          <div className="status-pill">{supportContract?.current_gate || 'Capabilities unavailable'}</div>
        </div>

        <div className="api-grid api-grid-polished api-capabilities-grid" aria-label="API capabilities support contract">
          <div className="api-card wide">
            <strong>Current gate</strong>
            {supportContract ? (
              <>
                <p><b>{supportContract.current_gate}</b></p>
                <p>{supportContract.support_policy}</p>
                <p>{supportContract.unsupported_policy}</p>
              </>
            ) : (
              <p>/api/capabilities is unavailable, so this frontend falls back to runtime health only and will not infer broad support from filenames or saved browser entries.</p>
            )}
          </div>

          <div className="api-card">
            <strong>Runtime readiness</strong>
            <p><b>loaded_now:</b> {loadedNow ? 'true' : 'false'}</p>
            <p><b>generation_ready:</b> {generationReady ? 'true' : 'false'}</p>
            <p><b>active_model_id:</b> {runtime?.active_model_id || 'none'}</p>
          </div>

          <div className="api-card">
            <strong>Supported quantization</strong>
            <p>{summarizeCapabilityItems(capabilities?.supported_quantization, 'Not advertised by this backend.')}</p>
          </div>

          <div className="api-card">
            <strong>Planned quantization</strong>
            <p>{summarizeCapabilityItems(capabilities?.planned_quantization, 'Not advertised by this backend.')}</p>
            <p>These lanes must keep returning typed errors until implementation and evidence land.</p>
          </div>

          <div className="api-card">
            <strong>Model family boundaries</strong>
            <p><b>Supported:</b> {summarizeCapabilityItems(capabilities?.supported_model_families, 'Not advertised by this backend.')}</p>
            <p><b>Planned:</b> {summarizeCapabilityItems(capabilities?.planned_model_families, 'Not advertised by this backend.')}</p>
          </div>

          <div className="api-card">
            <strong>Validated target</strong>
            {currentTarget ? (
              <>
                <code>{currentTarget.id}</code>
                <p>{formatCapabilityStatus(currentTarget.status)} · {currentTarget.family} · {currentTarget.quantization}</p>
                <p>{currentTarget.evidence}</p>
              </>
            ) : (
              <p>No compatibility rows advertised yet.</p>
            )}
          </div>

          <div className="api-card">
            <strong>Selected model contract</strong>
            {selectedModel ? (
              <>
                <code>{selectedModel.id}</code>
                <p><b>{compatibilityHintLabel(selectedCompatibilityHint, 'No matching row')}</b></p>
                <p>{selectedCompatibilitySupported ? compatibilityHintCopy(selectedCompatibilityHint) : `${compatibilityHintCopy(selectedCompatibilityHint)} Do not treat this selected model as chat-supported unless runtime readiness is also green.`}</p>
              </>
            ) : (
              <p>No selected model. Capability rows remain evidence boundaries, not a catalog of everything on disk.</p>
            )}
          </div>

          <div className="api-card wide">
            <strong>COMPATIBILITY.md rows mirrored from /api/capabilities</strong>
            {compatibilityTargets.length ? (
              <div className="api-feature-list capability-target-list">
                {compatibilityTargets.map((target) => (
                  <div key={target.id}>
                    <span>{target.id}</span>
                    <strong className={capabilityStatusTone(target.status)}>{formatCapabilityStatus(target.status)} · {target.family} · {target.quantization}</strong>
                    <small>Metadata: {formatCapabilityStatus(target.metadata_parses)} · tokenizer: {formatCapabilityStatus(target.tokenizer_works)} · tensors: {formatCapabilityStatus(target.tensors_load)} · generation: {formatCapabilityStatus(target.generation_runs)} · frontend load: {formatCapabilityStatus(target.frontend_load_path_verified)}</small>
                    <small>Template: {formatCapabilityStatus(target.chat_template_shape_pack || 'not_started')} · 512-context: {formatCapabilityStatus(target.bounded_context_512_pack || 'not_started')} · 1024-context: {formatCapabilityStatus(target.bounded_context_1024_pack || 'not_started')} · perf: {formatCapabilityStatus(target.performance_measured || 'not_started')}</small>
                    <small>{target.next_step}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p>No compatibility rows advertised yet.</p>
            )}
          </div>

          <div className="api-card wide">
            <strong>Unsupported / partial API features</strong>
            {guardedFeatures.length ? (
              <div className="api-feature-list">
                {guardedFeatures.map((feature) => (
                  <div key={feature.id}>
                    <span>{feature.id}</span>
                    <strong className={capabilityStatusTone(feature.status)}>{formatCapabilityStatus(feature.status)}</strong>
                    <small>{guardedCapabilityCopy(feature, 'API affordances and frontend controls')}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p>No unsupported or partial API rows advertised.</p>
            )}
          </div>
        </div>
      </section>
    </section>
  )
}
