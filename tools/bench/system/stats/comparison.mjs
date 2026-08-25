export function analyzePairedSamples(samples, options = {}) {
  if (!Array.isArray(samples)) throw new TypeError('samples must be an array')
  const higherIsBetter = options.higherIsBetter ?? true
  const bootstrapSamples = options.bootstrapSamples ?? 5000
  const confidence = options.confidence ?? 0.95
  const seed = options.seed ?? 0x43414d45
  if (typeof higherIsBetter !== 'boolean') throw new TypeError('higherIsBetter must be a boolean')
  if (!Number.isSafeInteger(bootstrapSamples) || bootstrapSamples < 1) {
    throw new RangeError('bootstrapSamples must be a positive safe integer')
  }
  if (!Number.isFinite(confidence) || confidence <= 0 || confidence >= 1) {
    throw new RangeError('confidence must be between 0 and 1')
  }
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
    throw new RangeError('seed must be a u32 integer')
  }

  const logRatios = []
  const excludedPairs = []
  samples.forEach((sample, index) => {
    const reason = invalidPairReason(sample)
    if (reason) {
      excludedPairs.push({ index, reason })
      return
    }
    logRatios.push(Math.log(sample.head / sample.base))
  })

  if (logRatios.length === 0) {
    return {
      validPairs: 0,
      excludedPairs,
      medianRatioHeadOverBase: null,
      bootstrapCi: null,
      observedDirection: 'insufficient_data',
      verdict: 'INCONCLUSIVE_NOISE',
      seed,
      bootstrapSamples,
      confidence,
    }
  }

  logRatios.sort((left, right) => left - right)
  const medianLogRatio = median(logRatios)
  const interval = bootstrapMedianInterval(logRatios, {
    bootstrapSamples,
    confidence,
    seed,
  })
  const ratio = Math.exp(medianLogRatio)
  const ci = interval.map(Math.exp)

  return {
    validPairs: logRatios.length,
    excludedPairs,
    medianRatioHeadOverBase: ratio,
    bootstrapCi: ci,
    observedDirection: direction(ci, higherIsBetter, logRatios.length),
    // A practical regression margin is deliberately absent in Phase 1.
    verdict: 'INCONCLUSIVE_NOISE',
    seed,
    bootstrapSamples,
    confidence,
  }
}

export function median(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new RangeError('median requires a non-empty array')
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new TypeError('median values must be finite numbers')
  }
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

function bootstrapMedianInterval(values, { bootstrapSamples, confidence, seed }) {
  if (values.length === 1) return [values[0], values[0]]
  const random = mulberry32(seed)
  const medians = new Array(bootstrapSamples)
  const sample = new Array(values.length)
  for (let iteration = 0; iteration < bootstrapSamples; iteration += 1) {
    for (let index = 0; index < values.length; index += 1) {
      sample[index] = values[Math.floor(random() * values.length)]
    }
    medians[iteration] = median(sample)
  }
  medians.sort((left, right) => left - right)
  const tail = (1 - confidence) / 2
  return [percentile(medians, tail), percentile(medians, 1 - tail)]
}

function percentile(sorted, probability) {
  const position = (sorted.length - 1) * probability
  const low = Math.floor(position)
  const high = Math.min(low + 1, sorted.length - 1)
  const fraction = position - low
  return sorted[low] + (sorted[high] - sorted[low]) * fraction
}

function invalidPairReason(sample) {
  if (sample === null || typeof sample !== 'object' || Array.isArray(sample)) return 'pair is not an object'
  if (!Number.isFinite(sample.base) || sample.base <= 0) return 'base must be a positive finite number'
  if (!Number.isFinite(sample.head) || sample.head <= 0) return 'head must be a positive finite number'
  return null
}

function direction(ci, higherIsBetter, validPairs) {
  if (validPairs < 2) return 'insufficient_data'
  const [low, high] = ci
  if (low > 1) return higherIsBetter ? 'head_faster' : 'head_slower'
  if (high < 1) return higherIsBetter ? 'head_slower' : 'head_faster'
  return 'no_clear_direction'
}

function mulberry32(seed) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000
  }
}
