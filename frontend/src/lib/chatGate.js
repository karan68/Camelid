import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel } from './capabilities.js'
import { isRunnableInCurrentRuntime, modelRuntimeIdMatches } from './modelState.js'

/* The backend's exact-artifact verdict for this model's GGUF, from
   `/api/models/local`. `classify_model_lane()` requires BOTH an implemented
   `general.architecture` parsed from the real header AND
   `filename_is_supported_exact_row()` — an exact filename equality against the
   curated catalog whose row id must also be `supported_*`. That conjunction is
   strictly stronger evidence than matching a display name against a row id, and
   lib/modelLanes.js has treated it as authoritative for the Models page since the
   same problem was found there.

   It is required here for the same reason: a compatibility row id concatenates the
   model's `general.finetune` token that the release filename may omit
   (`qwen3_0_6b_instruct_q8_0` vs `Qwen3-0.6B-Q8_0.gguf`). Whether the identity match
   could see that token depended on which path issued the load — the engine's startup
   auto-load names a model from GGUF metadata, `POST /api/models/load` names it
   whatever id the caller sent — so one file produced two contradictory claims
   ("Local chat ready" vs "unverified, no parity guarantee") on the same machine.

   It only ever ADDS the contract's own verdict for an exact artifact; it cannot
   unlock chat by itself, because `chatUnlocked` still requires runtime readiness. */
function backendMarksSupportedRow(model) {
  return model?.lane_class === 'supported'
}

const GEMMA4_26B_LANE_SCOPED_ROW = 'gemma4_26b_a4b_it_q4_0'
const GEMMA4_SERVE_LANES = new Set(['ghost_moe', 'local', 'distributed', 'cuda'])

function normalizedGemma4ServeLane(runtime) {
  const lane = String(runtime?.gemma4_serve_lane || '').trim().toLowerCase().replace(/-/g, '_')
  return GEMMA4_SERVE_LANES.has(lane) ? lane : null
}

function isGemma426bLaneScopedRow(model, hint) {
  if (hint?.kind === 'compatibility' && hint?.exact === true
    && hint?.target?.id === GEMMA4_26B_LANE_SCOPED_ROW) return true
  const identities = [
    model?.catalog_id,
    model?.compatibility_id,
    model?.runtime_model_name,
    model?.id,
    model?.name,
    model?.model_path,
    model?.hf_filename,
  ]
  return identities.some((identity) => {
    const normalized = String(identity || '').toLowerCase().replace(/[^a-z0-9]+/g, '_')
    return normalized === GEMMA4_26B_LANE_SCOPED_ROW
      || (/gemma_?4_26b(?:_|$)/.test(normalized) && normalized.includes('q4_0'))
  })
}

function supportedCatalogGhostCuda(runtime, runtimeLane) {
  return runtimeLane === 'ghost_moe'
    && runtime?.backend === 'gemma4-runtime'
    && runtime?.gemma4_ghost_catalog_managed === true
    && runtime?.gemma4_ghost_backend === 'cuda'
    && runtime?.gemma4_ghost_common_gpu_active === true
    && runtime?.gemma4_ghost_experts_gpu_active === true
    && runtime?.gemma4_ghost_head_gpu_active === true
}

function runtimeLaneHint(hint, laneScopedRow, runtimeLane) {
  const reason = runtimeLane === 'ghost_moe'
    ? 'This Ghost-MoE run is outside the supported catalog-managed Windows CUDA lane, or one of its GPU components is inactive.'
    : 'This row has no recognized distributed or catalog-managed Windows CUDA Ghost runtime report, so this run remains experimental.'
  const target = hint?.target || (laneScopedRow
    ? {
        id: GEMMA4_26B_LANE_SCOPED_ROW,
        family: 'gemma4_a4b_moe_decoder',
        quantization: 'Q4_0',
      }
    : null)
  if (!target) return hint
  return {
    ...(hint || { kind: 'runtime_lane_mismatch', exact: false }),
    kind: 'runtime_lane_mismatch',
    confidence: 'experimental runtime lane is outside the row support scope',
    target: {
      ...target,
      status: 'experimental_runtime_lane',
      evidence: reason,
      next_step: reason,
    },
  }
}

export function getChatGateState(capabilities, model, runtime) {
  const runtimeLoaded = Boolean(runtime?.loaded_now && modelRuntimeIdMatches(model, runtime))
  const runtimeGenerationReady = Boolean(runtime?.generation_ready && modelRuntimeIdMatches(model, runtime))
  const runtimeReady = Boolean(isRunnableInCurrentRuntime(model, runtime) && runtimeLoaded && runtimeGenerationReady)
  const discoveredHint = findCompatibilityHint(capabilities, model)
  const runtimeLane = normalizedGemma4ServeLane(runtime)
  const laneScopedRow = isGemma426bLaneScopedRow(model, discoveredHint)
  const scopedTarget = laneScopedRow
    ? capabilities?.model_compatibility?.find((row) => row?.id === GEMMA4_26B_LANE_SCOPED_ROW)
    : null
  const contractHint = scopedTarget
    ? {
        kind: 'compatibility',
        target: scopedTarget,
        confidence: 'exact Gemma 4 26B artifact identity',
        exact: true,
      }
    : discoveredHint
  // Support evidence is lane-scoped. The 26B row is green only for distributed
  // serve or the durable catalog-managed Windows CUDA Ghost pair with every GPU
  // component live. Ad-hoc/CPU/Metal/partial Ghost runs fail closed.
  const runtimeLaneEligible = !laneScopedRow
    || runtimeLane === 'distributed'
    || supportedCatalogGhostCuda(runtime, runtimeLane)
  // Once /api/models/local supplies a lane verdict, it owns artifact identity.
  // Falling back to name matching after an explicit experimental verdict would
  // let a same-named, wrong-hash file inherit the supported contract. Runtime
  // lane scope is an additional gate: a supported exact row cannot promote an
  // ad-hoc or partially accelerated Ghost run.
  const hasBackendLaneVerdict = Boolean(model?.lane_class)
  const artifactSupported = hasBackendLaneVerdict
    ? backendMarksSupportedRow(model)
    : isCompatibilitySupportedForModel(capabilities, model)
  const contractSupported = Boolean(runtimeLaneEligible && artifactSupported)
  const hint = runtimeLaneEligible
    ? contractHint
    : runtimeLaneHint(contractHint, laneScopedRow, runtimeLane)
  const chatUnlocked = Boolean(runtimeReady && contractSupported)
  // Experimental lane: the model loaded and is generation-ready (so its architecture
  // is implemented — generation_ready is false for unimplemented archs) but it is NOT
  // a supported contract row. A separate, weaker affordance from the supported gate:
  // chat is allowed but every turn is marked unverified with no parity claim.
  const experimentalUnlocked = Boolean(runtimeReady && !contractSupported)
  const chatMode = contractSupported ? 'supported' : experimentalUnlocked ? 'experimental' : 'blocked'

  return {
    hint,
    runtimeReady,
    runtimeLoaded,
    runtimeGenerationReady,
    contractSupported,
    chatUnlocked,
    experimentalUnlocked,
    chatMode,
    label: compatibilityHintLabel(hint, 'No matching COMPATIBILITY.md row'),
    copy: compatibilityHintCopy(hint),
  }
}
