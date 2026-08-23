import { validateComparison, validatePlan, validateRuntimeSample } from './lib/contracts.mjs'
import { analyzePairedSamples, median } from './stats/comparison.mjs'

const METRICS = new Map([
  ['tokens_per_second', { higherIsBetter: true }],
  ['load_ms', { higherIsBetter: false }],
  ['prefill_ms', { higherIsBetter: false }],
  ['ttft_ms', { higherIsBetter: false }],
  ['decode_ms', { higherIsBetter: false }],
  ['peak_rss_bytes', { higherIsBetter: false }],
])

export function buildComparison(plan, samples, options = {}) {
  validatePlan(plan)
  if (!Array.isArray(samples)) throw new TypeError('samples must be an array')
  samples.forEach(validateRuntimeSample)
  requireBaseHead(plan)
  const bootstrapSamples = options.bootstrapSamples ?? 5000
  const baseSeed = options.seed ?? 0x43414d45
  const runtime = []

  for (const workload of plan.workloads) {
    for (const metric of workload.primary_metrics) {
      const policy = METRICS.get(metric)
      if (!policy) throw new Error(`unsupported comparison metric ${metric}`)
      const validPairs = []
      const excludedPairs = []
      const workloadSamples = samples.filter((sample) => sample.workload_id === workload.id)
      for (let processBlock = 0; processBlock < workload.repetitions; processBlock += 1) {
        const base = workloadSamples.find((sample) => sample.process_block === processBlock && sample.arm_id === 'base')
        const head = workloadSamples.find((sample) => sample.process_block === processBlock && sample.arm_id === 'head')
        const reason = excludedReason(base, head, metric)
        if (reason) {
          excludedPairs.push({ process_block: processBlock, reason })
          continue
        }
        validPairs.push({ base: base.metrics[metric], head: head.metrics[metric] })
      }

      const seed = mixSeed(baseSeed, `${workload.id}:${metric}`)
      const analysis = analyzePairedSamples(validPairs, {
        higherIsBetter: policy.higherIsBetter,
        bootstrapSamples,
        confidence: 0.95,
        seed,
      })
      const invalidVerdict = invalidityVerdict(workloadSamples)
      runtime.push({
        workload_id: workload.id,
        metric,
        valid_pairs: analysis.validPairs,
        excluded_pairs: excludedPairs,
        base_median: validPairs.length > 0 ? median(validPairs.map((pair) => pair.base)) : null,
        head_median: validPairs.length > 0 ? median(validPairs.map((pair) => pair.head)) : null,
        median_ratio_head_over_base: analysis.medianRatioHeadOverBase,
        bootstrap_ci95: analysis.bootstrapCi,
        practical_margin: null,
        observed_direction: analysis.observedDirection,
        bootstrap_seed: seed,
        bootstrap_samples: bootstrapSamples,
        verdict: invalidVerdict ?? 'INCONCLUSIVE_NOISE',
      })
    }
  }

  const comparison = {
    schema: 'camelid.benchmark.comparison/v1',
    campaign_id: plan.campaign_id,
    runtime,
    agents: [],
  }
  validateComparison(comparison)
  return comparison
}

function requireBaseHead(plan) {
  const ids = plan.source_arms.map((arm) => arm.id).sort()
  if (ids.length !== 2 || ids[0] !== 'base' || ids[1] !== 'head') {
    throw new Error('Phase 1 comparison requires exactly two arms named base and head')
  }
}

function excludedReason(base, head, metric) {
  if (!base) return 'base sample is missing'
  if (!head) return 'head sample is missing'
  if (base.validity !== 'valid') return `base sample is ${base.validity}: ${base.invalid_reason}`
  if (head.validity !== 'valid') return `head sample is ${head.validity}: ${head.invalid_reason}`
  const baseValue = base.metrics?.[metric]
  const headValue = head.metrics?.[metric]
  if (!Number.isFinite(baseValue) || baseValue <= 0) return `base ${metric} is not a positive finite number`
  if (!Number.isFinite(headValue) || headValue <= 0) return `head ${metric} is not a positive finite number`
  return null
}

function invalidityVerdict(samples) {
  const validities = new Set(samples.map((sample) => sample.validity))
  if (validities.has('invalid_correctness')) return 'INVALID_CORRECTNESS'
  if (validities.has('invalid_backend')) return 'INVALID_BACKEND'
  if (validities.has('unsupported')) return 'UNSUPPORTED'
  if ([...validities].some((validity) => validity !== 'valid')) return 'INVALID_INFRASTRUCTURE'
  return null
}

function mixSeed(seed, text) {
  let hash = seed >>> 0
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash
}
