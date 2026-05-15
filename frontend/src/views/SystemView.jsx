import { capabilityStatusTone, displayCapabilityCopy, displayCapabilityId, findCompatibilityHint, formatCapabilityStatus, frontendSupportContractCopy, guardedCapabilityCopy, isExactCompatibilityHint, isGuardedCapabilityStatus, isSupportedCapabilityStatus } from '../lib/capabilities'
import { describeModelState, getRuntimeRequestModelId } from '../lib/modelState'

function runtimeReadinessLabel(runtime) {
  if (runtime?.generation_ready) return 'Loaded for local generation'
  if (runtime?.loaded_now) return 'Loaded, checking generation readiness'
  return 'Waiting for a generation-ready model'
}

export default function SystemView({ runtime, selectedModel, capabilities }) {
  const runtimePill = runtimeReadinessLabel(runtime)
  const selectedModelName = selectedModel?.name || 'No next-chat model selected'
  const apiBase = runtime?.api_base || 'Local API unavailable'
  const modelId = getRuntimeRequestModelId(selectedModel, runtime, '<loaded-model-id>') || '<loaded-model-id>'
  const supportContract = capabilities?.support_contract
  const supportContractCurrentGate = frontendSupportContractCopy(capabilities)
  const compatibilityTargets = capabilities?.model_compatibility || []
  const apiFeatures = capabilities?.api_features || []
  const supportedFeatures = apiFeatures.filter((feature) => isSupportedCapabilityStatus(feature.status))
  const unsupportedFeatures = apiFeatures.filter((feature) => isGuardedCapabilityStatus(feature.status))
  const selectedCompatibilityHint = findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = isExactCompatibilityHint(selectedCompatibilityHint) ? selectedCompatibilityHint.target : null
  const exactRowQuantEvidence = compatibilityTargets.length
    ? compatibilityTargets.map((target) => `${target.id}: ${target.quantization} (${formatCapabilityStatus(target.status)})`).join(' · ')
    : 'No exact compatibility rows advertised.'
  const exactRowFamilyEvidence = compatibilityTargets.length
    ? compatibilityTargets.map((target) => `${target.id}: ${target.family} (${formatCapabilityStatus(target.status)})`).join(' · ')
    : 'No exact compatibility rows advertised.'

  return (
    <section className="view-stack system-layout-single view-shell">
      <div className="panel panel-hero system-hero system-hero-separated">
        <div className="view-hero-copy">
          <p className="panel-kicker">System</p>
          <h2>Runtime, readiness, and local API access</h2>
          <p className="hero-summary">This is the operational view. It keeps runtime health, model readiness, and developer connection details in one calmer place while Library stays focused on browsing and setup.</p>
        </div>
        <div className="view-hero-stats system-hero-pills system-hero-pills-polished">
          <div className={`status-pill ${runtime?.generation_ready ? 'ready' : 'warm'}`}>{runtimePill}</div>
          <div className="status-pill">{apiBase}</div>
        </div>
      </div>

      <div className="runtime-grid runtime-grid-polished">
        <div className="panel panel-section">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Runtime</p>
              <h2>What Camelid can do right now</h2>
            </div>
            <p className="model-summary">A quick operational summary of the local engine, loaded model, and what is actually ready for local completions.</p>
          </div>
          <div className="runtime-stat-grid">
            <div className="runtime-stat"><span>Runtime state</span><strong>{runtime?.generation_ready ? 'Generation-ready' : runtime?.loaded_now ? 'Loaded, not generation-ready' : 'No generation-ready model'}</strong></div>
            <div className="runtime-stat"><span>Local engine</span><strong>{runtime?.engine || 'Unknown'}</strong></div>
            <div className="runtime-stat"><span>Loaded model</span><strong>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</strong></div>
            <div className="runtime-stat"><span>Generation ready</span><strong>{runtime?.generation_ready ? 'Yes' : 'No'}</strong></div>
            <div className="runtime-stat"><span>Acceleration</span><strong>CPU path today; optimized kernels/GPU are not wired yet</strong></div>
            <div className="runtime-stat"><span>Next chat selection</span><strong>{selectedModelName}</strong></div>
            <div className="runtime-stat"><span>API base</span><strong>{apiBase}</strong></div>
          </div>
        </div>

        <div className="panel panel-section">
          <div className="section-heading">
            <div>
              <p className="panel-kicker">Capability</p>
              <h2>What Camelid is handling locally</h2>
            </div>
            <p className="model-summary">A plain-language snapshot of what is already available without reaching outside this machine.</p>
          </div>
          <div className="activity-feed activity-feed-polished">
            <div className="activity-item">Persistent conversations are already available from local storage.</div>
            <div className="activity-item">Saved memory remains on-device and can be recalled in later chats.</div>
            <div className="activity-item">Camelid is using the local CPU generation path today; GPU acceleration remains future work.</div>
            <div className="activity-item">Current next-chat model state: {describeModelState(selectedModel)}</div>
            <div className="activity-item">Chat stays blocked until Camelid reports generation_ready for the selected model; once ready, chat runs until EOS, an explicit request limit, or the backend context window.</div>
            <div className="activity-item">The standard /v1-compatible local API is exposed at {apiBase}.</div>
          </div>
        </div>
      </div>

      <section className="panel api-panel panel-section">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Developer</p>
            <h2>Standard /v1-compatible local API</h2>
            <p className="hero-summary">Use the same local runtime through standard endpoints for apps, scripts, and quick terminal checks.</p>
          </div>
          <div className={`status-pill ${runtime?.generation_ready ? 'ready' : 'warm'}`}>{runtime?.generation_ready ? 'Local /v1 generation ready' : runtime?.loaded_now ? 'Local /v1 loaded, not ready' : 'Load a model to use generation'}</div>
        </div>
        <div className="api-grid api-grid-polished">
          <div className="api-card">
            <strong>Chat completions</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/chat/completions` : 'Unavailable until the local API is running'}</code>
            <p>Runs only after a local GGUF is loaded and Camelid reports generation_ready=true.</p>
          </div>
          <div className="api-card">
            <strong>Models</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/models` : 'Unavailable until the local API is running'}</code>
            <p>Lists the currently loaded runtime model; this is not a broad model catalog.</p>
          </div>
          <div className="api-card">
            <strong>Health</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/health` : 'Unavailable until the local API is running'}</code>
            <p>Source of truth for active_model_id and generation_ready.</p>
          </div>
          <div className="api-card">
            <strong>Capabilities</strong>
            <code>{runtime?.api_base ? `${runtime.api_base}/api/capabilities` : 'Unavailable until the local API is running'}</code>
            <p>Support contract for model families, quantization, API features, and compatibility rows.</p>
          </div>
          <div className="api-card wide api-card-code">
            <strong>Readiness-gated curl</strong>
            <pre>{runtime?.api_base ? `# Use only after /v1/health returns generation_ready=true\ncurl ${runtime.api_base}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello from Camelid"}],
    "temperature": 0
  }'` : 'Start the local runtime to see a ready-to-copy curl example.'}</pre>
          </div>
        </div>
      </section>

      <section className="panel api-panel panel-section">
        <div className="panel-header-row panel-header-row-wide">
          <div>
            <p className="panel-kicker">Support contract</p>
            <h2>Evidence-based compatibility</h2>
            <p className="hero-summary">This mirrors /api/capabilities so the UI does not imply unvalidated model families, quantization formats, or API features.</p>
          </div>
          <div className="status-pill">{supportContractCurrentGate}</div>
        </div>

        <div className="api-grid api-grid-polished api-capabilities-grid" aria-label="Capabilities support contract">
          <div className="api-card wide">
            <strong>Current contract</strong>
            {supportContract ? (
              <>
                <p><b>Current gate:</b> {supportContractCurrentGate}</p>
                <p>{supportContract.support_policy}</p>
                <p>{supportContract.unsupported_policy}</p>
              </>
            ) : (
              <p>/api/capabilities is unavailable, so Camelid falls back to health/model readiness only and does not infer broader support.</p>
            )}
          </div>

          <div className="api-card">
            <strong>Selected exact-row evidence</strong>
            {selectedCompatibilityTarget ? (
              <>
                <code>{selectedCompatibilityTarget.id}</code>
                <p>{formatCapabilityStatus(selectedCompatibilityTarget.status)} · {selectedCompatibilityTarget.family} · {selectedCompatibilityTarget.quantization}</p>
                <p><b>Readiness gate:</b> {displayCapabilityCopy(selectedCompatibilityTarget.frontend_readiness_gate || 'not advertised')}</p>
                <p>{displayCapabilityCopy(selectedCompatibilityTarget.evidence || selectedCompatibilityTarget.next_step || 'No row evidence advertised.')}</p>
              </>
            ) : (
              <p>No selected model exact row matched, so System will not promote runtime health, families, or quant lists into support.</p>
            )}
          </div>

          <div className="api-card">
            <strong>Exact-row quant evidence</strong>
            <p>{exactRowQuantEvidence}</p>
            <p>Quant labels here come only from compatibility rows and do not unlock chat without a matching loaded model.</p>
          </div>

          <div className="api-card">
            <strong>Exact-row family evidence</strong>
            <p>{exactRowFamilyEvidence}</p>
            <p>Family labels remain row-scoped evidence boundaries, not broad support for neighboring files.</p>
          </div>

          <div className="api-card">
            <strong>Validated API features</strong>
            {supportedFeatures.length ? (
              <>
                {supportedFeatures.map((feature) => (
                  <p key={feature.id}><b>{displayCapabilityId(feature.id)}:</b> {displayCapabilityCopy(feature.notes)}</p>
                ))}
              </>
            ) : (
              <p>No supported API feature rows advertised yet.</p>
            )}
          </div>

          <div className="api-card wide">
            <strong>COMPATIBILITY.md rows from /api/capabilities</strong>
            {compatibilityTargets.length ? (
              <div className="api-feature-list capability-target-list">
                {compatibilityTargets.map((target) => (
                  <div key={target.id}>
                    <span>{target.family} · {target.quantization}</span>
                    <strong className={capabilityStatusTone(target.status)}>{target.id}: {formatCapabilityStatus(target.status)}</strong>
                    <small>Metadata: {formatCapabilityStatus(target.metadata_parses)} · tokenizer: {formatCapabilityStatus(target.tokenizer_works)} · tensors: {formatCapabilityStatus(target.tensors_load)} · generation: {formatCapabilityStatus(target.generation_runs)}</small>
                    <small>{displayCapabilityCopy(target.next_step)}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p>No compatibility rows advertised yet.</p>
            )}
          </div>

          <div className="api-card wide">
            <strong>Unsupported / partial API features</strong>
            {unsupportedFeatures.length ? (
              <div className="api-feature-list">
                {unsupportedFeatures.map((feature) => (
                  <div key={feature.id}>
                    <span>{displayCapabilityId(feature.id)}</span>
                    <strong className={capabilityStatusTone(feature.status)}>{formatCapabilityStatus(feature.status)}</strong>
                    <small>{displayCapabilityCopy(guardedCapabilityCopy(feature, 'System/API controls'))}</small>
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
