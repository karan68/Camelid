export const LLAMA_CACHE_TYPES = new Set([
  'f32',
  'f16',
  'bf16',
  'q8_0',
  'q4_0',
  'q4_1',
  'iq4_nl',
  'q5_0',
  'q5_1',
])

export function parseArgs(argv) {
  const args = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const value = inline ?? (argv[i + 1]?.startsWith('--') ? 'true' : argv[++i] ?? 'true')
    args.set(key, value)
  }
  return args
}

export function requestMode(args) {
  const mode = args.has('chat') ? 'chat' : (args.get('request-mode') || 'raw')
  if (!['raw', 'chat'].includes(mode)) {
    throw new Error(`--request-mode must be raw or chat, got ${JSON.stringify(mode)}`)
  }
  return mode
}

export function camelidLaunchConfig(args, baseEnv = process.env) {
  const lane = args.get('camelid-lane') || 'cpu'
  if (!['cpu', 'metal'].includes(lane)) {
    throw new Error(`--camelid-lane must be cpu or metal, got ${JSON.stringify(lane)}`)
  }
  const env = { ...baseEnv, CUDA_VISIBLE_DEVICES: '-1' }
  if (lane === 'metal') {
    return {
      lane,
      env: { ...env, CAMELID_LFM2_METAL: '1' },
      serveArgs: ['--gpu', 'on'],
    }
  }
  return {
    lane,
    env: { ...env, CAMELID_LFM2_METAL: '0' },
    serveArgs: ['--gpu', 'off', '--deterministic'],
  }
}

export function positiveInt(value, label) {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed < 1 || String(parsed) !== String(value).trim()) {
    throw new Error(`${label} must be a positive integer, got ${JSON.stringify(value)}`)
  }
  return parsed
}

export function oracleConfig(args) {
  const cacheTypeK = args.get('llama-cache-type-k') || 'f32'
  const cacheTypeV = args.get('llama-cache-type-v') || 'f32'
  const flashAttn = args.get('llama-flash-attn') || 'off'
  const repack = args.get('llama-repack') || 'off'
  if (!LLAMA_CACHE_TYPES.has(cacheTypeK)) {
    throw new Error(`unsupported --llama-cache-type-k ${JSON.stringify(cacheTypeK)}`)
  }
  if (!LLAMA_CACHE_TYPES.has(cacheTypeV)) {
    throw new Error(`unsupported --llama-cache-type-v ${JSON.stringify(cacheTypeV)}`)
  }
  if (!['on', 'off', 'auto'].includes(flashAttn)) {
    throw new Error(`--llama-flash-attn must be on, off, or auto, got ${JSON.stringify(flashAttn)}`)
  }
  if (!['on', 'off'].includes(repack)) {
    throw new Error(`--llama-repack must be on or off, got ${JSON.stringify(repack)}`)
  }
  return { cacheTypeK, cacheTypeV, flashAttn, repack }
}

export function buildLongPrompt(targetTokens) {
  const paragraph =
    'The river wound past the old stone mill, where farmers once ground wheat into flour for the village bakeries. ' +
    'Each autumn the orchards filled with ripe apples, and children carried baskets along the dusty lane toward the market square. ' +
    'A blacksmith hammered iron near the well while merchants argued over the price of wool and salt. '
  const repetitions = Math.ceil((targetTokens * 3.6) / paragraph.length)
  return paragraph.repeat(repetitions).slice(0, Math.floor(targetTokens * 3.6))
}

export function buildMessages(prompt, systemMessage) {
  const messages = []
  if (systemMessage) messages.push({ role: 'system', content: systemMessage })
  messages.push({ role: 'user', content: prompt })
  return messages
}

export function buildGenerationRequest({ mode, prompt, messages, maxGen }) {
  const common = {
    max_tokens: maxGen,
    temperature: 0,
    stream: false,
    camelid_receipt: true,
  }
  return mode === 'chat' ? { ...common, messages } : { ...common, prompt }
}

export function actualTokenGate(actual, expectedRaw) {
  if (expectedRaw === undefined) {
    return { enabled: false, expected: null, actual, pass: null }
  }
  const expected = positiveInt(expectedRaw, '--expect-actual-tokens')
  return { enabled: true, expected, actual, pass: actual === expected }
}

export function requireNativeReceipt(response, mode) {
  const receipt = response?.camelid_receipt
  if (!receipt) {
    throw new Error(
      `native ${mode} response carried no camelid_receipt (do not synthesize one): ` +
      JSON.stringify(response?.error || response).slice(0, 300),
    )
  }
  const promptIds = receipt.result?.prompt_token_ids
  if (!Array.isArray(promptIds) || promptIds.length === 0) {
    throw new Error('native camelid_receipt carried no result.prompt_token_ids')
  }
  if (typeof receipt.result?.generated_text !== 'string') {
    throw new Error('native camelid_receipt carried no string result.generated_text')
  }
  return { receipt, promptIds, generatedText: receipt.result.generated_text }
}

export function verifierArgs({
  receiptPath,
  gguf,
  llamaServer,
  llamaCtx,
  llamaPort,
  verifyMode,
  oracle,
}) {
  const args = [
    'verify-receipt', receiptPath, '--gguf', gguf, '--llama-server', llamaServer,
    '--llama-ctx', String(llamaCtx), '--llama-port', String(llamaPort),
    '--llama-cache-type-k', oracle.cacheTypeK,
    '--llama-cache-type-v', oracle.cacheTypeV,
    '--llama-flash-attn', oracle.flashAttn,
  ]
  if (oracle.repack === 'off') args.push('--llama-no-repack')
  if (verifyMode === 'reference-only') args.push('--reference-only')
  if (verifyMode === 'self-only') args.push('--self-only')
  return args
}
