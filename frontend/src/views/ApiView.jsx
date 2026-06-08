import { capabilityStatusTone, displayCapabilityCopy, displayCapabilityId, exactRowSupportLanes, findCompatibilityHint, formatCapabilityStatus, frontendSupportContractCopy, guardedCapabilityCopy, isExactCompatibilityHint, isGuardedCapabilityStatus, isSupportedCapabilityStatus, rowSupportBoundaryCopy, rowSupportNextStepCopy } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { getRuntimeRequestModelId, modelRuntimeIdMatches } from '../lib/modelState'
import { StatusDot } from '../components/ui/StatusDot'
import { IconApi } from '../components/ui/icons'

function guardedApiFeatures(features = []) {
  return features.filter((feature) => isGuardedCapabilityStatus(feature.status))
}

function summarizeExactRowField(targets = [], field, fallback = 'No exact compatibility rows advertised by this backend.') {
  const rows = targets
    .filter((target) => target?.id && target?.[field])
    .map((target) => `${displayCapabilityCopy(target[field])}: ${formatCapabilityStatus(target.status)} (${target.id})`)
  return rows.length ? rows.join(' · ') : fallback
}

function supportLaneTitle(lane) {
  if (lane.key === 'template') return 'Template/Jinja readiness'
  if (lane.key === 'context') return 'Checked context readiness'
  return 'Throughput readiness'
}

export default function ApiView({ runtime, selectedModel, capabilities }) {
  const apiBase = runtime?.api_base || ''
  const modelId = getRuntimeRequestModelId(selectedModel, runtime, '<loaded-model-id>') || '<loaded-model-id>'
  const supportContract = capabilities?.support_contract
  const supportContractCurrentGate = frontendSupportContractCopy(capabilities)
  const compatibilityTargets = capabilities?.model_compatibility || []
  const apiFeatures = capabilities?.api_features || []
  const supportedFeatures = apiFeatures.filter((feature) => isSupportedCapabilityStatus(feature.status))
  const guardedFeatures = guardedApiFeatures(apiFeatures)
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = isExactCompatibilityHint(selectedCompatibilityHint) ? selectedCompatibilityHint.target : null
  const selectedCompatibilitySupported = selectedChatGate.contractSupported
  const selectedSupportLanes = exactRowSupportLanes(selectedCompatibilityTarget, apiFeatures)
  const generationReady = Boolean(runtime?.generation_ready)
  const loadedNow = Boolean(runtime?.loaded_now)
  const selectedRuntimeMatches = modelRuntimeIdMatches(selectedModel, runtime)
  const q8Runtime = runtime?.q8_runtime
  const selectedExactRowReady = selectedChatGate.chatUnlocked
  const readinessPillCopy = selectedExactRowReady
    ? 'Selected exact row ready'
    : generationReady && selectedModel && !selectedRuntimeMatches
      ? 'Different loaded model is ready'
      : generationReady
        ? 'Generation ready; exact row required'
        : 'Load a generation-ready exact row'
  const chatCompletionsCopy = selectedExactRowReady
    ? 'Runnable now for this selected GGUF because runtime readiness and the exact supported row both match.'
    : selectedCompatibilityTarget
      ? 'Keep UX chat gated until this selected exact row is loaded_now=true, generation_ready=true, and active_model_id matches.'
      : 'Keep UX chat gated; no selected exact compatibility row is available to pair with runtime readiness.'
  const curlExample = selectedExactRowReady
    ? `# Selected exact row is runtime-ready now\ncurl ${apiBase}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "${modelId}",\n    "messages": [{"role": "user", "content": "Hello from Camelid"}],\n    "temperature": 0\n  }'`
    : `# Blocked for UX chat until selected exact row evidence and runtime readiness both match\n# loaded_now=${loadedNow ? 'true' : 'false'} generation_ready=${generationReady ? 'true' : 'false'} active_model_id=${runtime?.active_model_id || 'none'}\n# selected_exact_row=${selectedCompatibilityTarget?.id || 'none'}`

  const runtimeStat = generationReady ? 'Ready' : loadedNow ? 'Loaded' : 'Offline'
  const runtimeTone = generationReady ? 'ready' : loadedNow ? 'warn' : 'neutral'
  const headerStatus = generationReady ? 'Generation-ready' : loadedNow ? 'Loaded, chat blocked' : 'No generation-ready model'
  const selectedRowStat = selectedExactRowReady ? 'Ready' : selectedCompatibilityTarget ? 'Gated' : 'None'
  const selectedRowSub = selectedCompatibilitySupported ? 'exact row supported' : selectedCompatibilityTarget ? 'matched, gated' : 'no exact row'

  return (
    <section className="api-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconApi size={14} /> API</p>
          <h1>Local API contract &amp; readiness</h1>
          <p className="cxv-sub">This view makes the backend support contract explicit: /api/capabilities describes what Camelid has evidence for, while /v1/health decides whether the currently loaded model can actually chat.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={runtimeTone} pulse={generationReady} label={headerStatus} />
        </div>
      </header>

      <div className="cxv-stat-grid">
        <div className="cxv-stat"><span>Runtime</span><strong>{runtimeStat}</strong><small>{generationReady ? 'generation_ready=true' : loadedNow ? 'loaded_now=true' : 'no model loaded'}</small></div>
        <div className="cxv-stat"><span>Loaded model</span><strong>{loadedNow ? 'Active' : 'None'}</strong><small title={runtime?.active_model_id || 'nothing loaded'}>{runtime?.active_model_id || 'nothing loaded'}</small></div>
        <div className="cxv-stat"><span>Selected row</span><strong>{selectedRowStat}</strong><small>{selectedRowSub}</small></div>
        <div className="cxv-stat"><span>Local API</span><strong>{apiBase ? 'Online' : 'Offline'}</strong><small>{apiBase || 'unavailable'}</small></div>
      </div>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Standard /v1-compatible surface</h2>
          <StatusDot tone={selectedExactRowReady ? 'ready' : 'warn'} label={readinessPillCopy} />
        </div>
        <p className="cxv-sub">Generation endpoints stay useful only when runtime readiness is green and the selected local GGUF has an exact supported compatibility row. Capability rows explain supported and guarded lanes, but they never override loaded_now/generation_ready or active_model_id matching.</p>
        <div className="sys-endpoints">
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Chat completions</strong><span className="cxv-tag">POST</span></div>
            <code>{apiBase ? `${apiBase}/v1/chat/completions` : 'Unavailable until the local API is running'}</code>
            <p>{chatCompletionsCopy}</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Model listing</strong><span className="cxv-tag">GET</span></div>
            <code>{apiBase ? `${apiBase}/v1/models` : 'Unavailable until the local API is running'}</code>
            <p>Lists the active runtime model. It is not a broad compatibility catalog.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Health</strong><span className="cxv-tag">GET</span></div>
            <code>{apiBase ? `${apiBase}/v1/health` : 'Unavailable until the local API is running'}</code>
            <p>Source of truth for active_model_id, loaded_now, and generation_ready.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Capabilities</strong><span className="cxv-tag">GET</span></div>
            <code>{apiBase ? `${apiBase}/api/capabilities` : 'Unavailable until the local API is running'}</code>
            <p>Support contract for exact compatibility rows, row-scoped family/quant evidence, API feature support, and typed guardrails.</p>
          </div>
        </div>
        <div className="sys-curl">
          <div className="sys-curl__head"><strong>Readiness-gated curl</strong><span className="cxv-tag">curl</span></div>
          <pre>{apiBase ? curlExample : 'Start the local runtime to see an exact-row readiness check.'}</pre>
        </div>
      </section>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>/api/capabilities summary</h2><span className="cxv-section__count">evidence boundaries</span></div>
        <p className="cxv-sub">The UI treats these rows as evidence boundaries, not marketing claims. Planned, partial, blocked, or unsupported rows remain visible but guarded.</p>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Current gate</strong>
            {supportContract ? (
              <>
                <p><b>{supportContractCurrentGate}</b></p>
                <p>{supportContract.support_policy}</p>
                <p>{supportContract.unsupported_policy}</p>
              </>
            ) : (
              <p>/api/capabilities is unavailable, so this frontend falls back to runtime health only and will not infer broad support from filenames or saved browser entries.</p>
            )}
          </div>

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Runtime readiness</strong>
            <p><b>loaded_now:</b> {loadedNow ? 'true' : 'false'}</p>
            <p><b>generation_ready:</b> {generationReady ? 'true' : 'false'}</p>
            <p><b>active_model_id:</b> {runtime?.active_model_id || 'none'}</p>
            <p><b>q8_policy:</b> {q8Runtime?.policy || 'unavailable'}</p>
            <p>{q8Runtime?.note || 'Q8 storage policy is reported by /v1/health when the runtime is online.'}</p>
          </div>
        </div>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Exact-row quant evidence</strong>
            <p>{summarizeExactRowField(compatibilityTargets, 'quantization')}</p>
            <p>Quant labels here come from compatibility rows only; broad quant lists do not unlock chat.</p>
          </div>
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Exact-row family evidence</strong>
            <p>{summarizeExactRowField(compatibilityTargets, 'family')}</p>
            <p>Family names remain row-scoped evidence boundaries, not inherited support for neighboring files.</p>
          </div>
        </div>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected exact-row evidence</strong>
            {selectedCompatibilityTarget ? (
              <>
                <code className="a-code">{selectedCompatibilityTarget.id}</code>
                <p>{formatCapabilityStatus(selectedCompatibilityTarget.status)} · {selectedCompatibilityTarget.family} · {selectedCompatibilityTarget.quantization}</p>
                <p><b>Scope:</b> {displayCapabilityCopy(selectedCompatibilityTarget.support_scope || 'not advertised')}</p>
                <p><b>Readiness gate:</b> {displayCapabilityCopy(selectedCompatibilityTarget.frontend_readiness_gate)}</p>
                <p><b>Latest checked:</b> {formatCapabilityStatus(selectedCompatibilityTarget.latest_checked_bucket)} · {formatCapabilityStatus(selectedCompatibilityTarget.latest_checked_result)}</p>
                <p><b>Latest output:</b> {displayCapabilityCopy(selectedCompatibilityTarget.latest_checked_output || 'not advertised')}</p>
                <p><b>Full-support status:</b> {formatCapabilityStatus(selectedCompatibilityTarget.full_support_status || 'not advertised')}</p>
                {selectedSupportLanes.map((lane) => (
                  <p key={lane.key}><b>{supportLaneTitle(lane)}:</b> {lane.label}. {displayCapabilityCopy(lane.copy)}</p>
                ))}
                <p><b>Remaining support boundary:</b> {displayCapabilityCopy(rowSupportBoundaryCopy(selectedCompatibilityTarget, apiFeatures))}</p>
                <p>{displayCapabilityCopy(selectedCompatibilityTarget.evidence)}</p>
              </>
            ) : (
              <p>No selected model exact row matched. This API view will not promote family names, saved paths, or runtime health into a support claim.</p>
            )}
          </div>

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected model contract</strong>
            {selectedModel ? (
              <>
                <code className="a-code">{selectedModel.id}</code>
                {selectedCompatibilityTarget ? (
                  <>
                    <p><b>{selectedCompatibilityTarget.id}: {formatCapabilityStatus(selectedCompatibilityTarget.status)}</b></p>
                    <p>{selectedCompatibilitySupported ? 'This selected model has an exact supported compatibility row; runtime readiness must still match before chat unlocks.' : 'An exact row matched, but it is not supported for chat at this gate.'}</p>
                  </>
                ) : (
                  <p>No exact compatibility row matched this selected model, so the API UI will not display family, quant-list, filename, or saved-path guesses as support evidence.</p>
                )}
              </>
            ) : (
              <p>No selected model. Capability rows remain evidence boundaries, not a catalog of everything on disk.</p>
            )}
          </div>
        </div>

        <div className="cxv-card cxv-card--flat sys-evidence">
          <strong>Supported API feature rows</strong>
          {supportedFeatures.length ? (
            <div className="sys-rows">
              {supportedFeatures.map((feature) => (
                <div key={feature.id} className="sys-row">
                  <div className="sys-row__head">
                    <span>{displayCapabilityId(feature.id)}</span>
                    <span className={statusTagClass(feature.status)}>{formatCapabilityStatus(feature.status)}</span>
                  </div>
                  <small>{displayCapabilityCopy(feature.notes || 'Advertised by /api/capabilities. These feature rows do not widen model support; chat still follows the selected exact-row and runtime readiness gate above.')}</small>
                </div>
              ))}
            </div>
          ) : (
            <p>No supported API feature rows advertised.</p>
          )}
        </div>
      </section>

      <details className="cxv-disclosure">
        <summary>Full compatibility evidence — every row and guarded feature from /api/capabilities</summary>
        <div className="cxv-disclosure__body">
          <div className="sys-rows-block">
            <strong>COMPATIBILITY.md rows mirrored from /api/capabilities</strong>
            {compatibilityTargets.length ? (
              <div className="sys-rows">
                {compatibilityTargets.map((target) => (
                  <div key={target.id} className="sys-row">
                    <div className="sys-row__head">
                      <span>{target.id}</span>
                      <span className={statusTagClass(target.status)}>{formatCapabilityStatus(target.status)} · {target.family} · {target.quantization}</span>
                    </div>
                    <small>Metadata: {formatCapabilityStatus(target.metadata_parses)} · tokenizer: {formatCapabilityStatus(target.tokenizer_works)} · tensors: {formatCapabilityStatus(target.tensors_load)} · generation: {formatCapabilityStatus(target.generation_runs)} · frontend load: {formatCapabilityStatus(target.frontend_load_path_verified)}</small>
                    <small>Template: {formatCapabilityStatus(target.chat_template_shape_pack || 'not_started')} · 512-context: {formatCapabilityStatus(target.bounded_context_512_pack || 'not_started')} · 1024-context: {formatCapabilityStatus(target.bounded_context_1024_pack || 'not_started')} · 2048-context: {formatCapabilityStatus(target.bounded_context_2048_pack || 'not_started')} · 4096-context: {formatCapabilityStatus(target.bounded_context_4096_pack || 'not_started')} · 8192-context: {formatCapabilityStatus(target.bounded_context_8192_pack || 'not_started')} · perf: {formatCapabilityStatus(target.performance_measured || 'not_started')}</small>
                    <small>{exactRowSupportLanes(target, apiFeatures).map((lane) => `${supportLaneTitle(lane).replace(' readiness', '')}: ${lane.label}`).join(' · ')}</small>
                    <small>{displayCapabilityCopy(rowSupportNextStepCopy(target, apiFeatures))}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="cxv-sub">No compatibility rows advertised yet.</p>
            )}
          </div>

          <div className="sys-rows-block">
            <strong>Unsupported / partial API features</strong>
            {guardedFeatures.length ? (
              <div className="sys-rows">
                {guardedFeatures.map((feature) => (
                  <div key={feature.id} className="sys-row">
                    <div className="sys-row__head">
                      <span>{displayCapabilityId(feature.id)}</span>
                      <span className={statusTagClass(feature.status)}>{formatCapabilityStatus(feature.status)}</span>
                    </div>
                    <small>{displayCapabilityCopy(guardedCapabilityCopy(feature, 'API affordances and frontend controls'))}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="cxv-sub">No unsupported or partial API rows advertised.</p>
            )}
          </div>
        </div>
      </details>
    </section>
  )
}

function statusTagClass(status) {
  const tone = capabilityStatusTone(status)
  if (tone === 'ready') return 'cxv-tag cxv-tag--ready'
  if (tone === 'warm') return 'cxv-tag cxv-tag--warn'
  return 'cxv-tag'
}
