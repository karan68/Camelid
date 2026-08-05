#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const base = (args.get('base') || 'http://127.0.0.1:8181').replace(/\/$/, '')
const model = args.get('model') || 'Bonsai-27B'
const imageArg = args.get('image') || null
const imagePath = imageArg ? resolve(imageArg) : null
const prompt = args.get('prompt') ||
  'Describe this image in exactly 24 short words. Do not stop early.'
const maxTokens = positiveInt(args.get('max-tokens') || '24', '--max-tokens')
const iterations = positiveInt(args.get('iterations') || '3', '--iterations')
const timeoutMs = positiveInt(args.get('timeout-ms') || '300000', '--timeout-ms')
const outputPath = args.get('out') ? resolve(args.get('out')) : null
const imageBytes = imagePath ? await readFile(imagePath) : null
const imageUrl = imagePath
  ? `data:image/${imagePath.toLowerCase().endsWith('.png') ? 'png' : 'jpeg'};base64,${imageBytes.toString('base64')}`
  : null

const runs = []
for (let iteration = 1; iteration <= iterations; iteration += 1) {
  const result = await measure(iteration)
  runs.push(result)
  process.stderr.write(
    `run ${iteration}: ttft=${result.ttft_ms.toFixed(3)} ms ` +
    `total=${result.total_ms.toFixed(3)} ms post_first=${result.post_first_ms.toFixed(3)} ms ` +
    `decode=${result.decode_tokens_per_second?.toFixed(3) ?? 'n/a'} tok/s ` +
    `events=${result.content_events} completion_tokens=${result.completion_tokens ?? 'n/a'}\n`,
  )
}

const report = {
  schema: 'camelid.prism_vision_stream_benchmark/v1',
  created_at: new Date().toISOString(),
  base,
  model,
  // Keep durable benchmark receipts portable and free of operator home paths.
  image: imageArg ? imageArg.replaceAll('\\', '/') : null,
  prompt,
  max_tokens: maxTokens,
  iterations,
  runs,
  summary: summarize(runs),
}

const json = `${JSON.stringify(report, null, 2)}\n`
if (outputPath) await writeFile(outputPath, json)
process.stdout.write(json)

async function measure(iteration) {
  const body = {
    model,
    messages: [{
      role: 'user',
      content: imageUrl ? [
          { type: 'text', text: prompt },
          { type: 'image_url', image_url: { url: imageUrl } },
        ] : prompt,
    }],
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: true,
    stream_options: { include_usage: true },
    camelid_enable_thinking: false,
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const started = performance.now()
  let firstContentAt = null
  let contentEvents = 0
  let generatedText = ''
  let usage = null
  try {
    const response = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 1000)}`)
    }
    if (!response.body) throw new Error('response has no streaming body')

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() || ''
      for (const line of lines) consumeSseLine(line)
    }
    buffer += decoder.decode()
    for (const line of buffer.split(/\r?\n/)) consumeSseLine(line)
  } finally {
    clearTimeout(timer)
  }

  const ended = performance.now()
  if (firstContentAt === null) throw new Error(`iteration ${iteration} produced no content`)
  const postFirstMs = ended - firstContentAt
  const postFirstTokens = usage?.completion_tokens == null
    ? null
    : Math.max(usage.completion_tokens - 1, 0)
  return {
    iteration,
    ttft_ms: firstContentAt - started,
    total_ms: ended - started,
    post_first_ms: postFirstMs,
    post_first_tokens: postFirstTokens,
    decode_tokens_per_second: postFirstTokens == null || postFirstMs <= 0
      ? null
      : postFirstTokens * 1000 / postFirstMs,
    content_events: contentEvents,
    completion_tokens: usage?.completion_tokens ?? null,
    prompt_tokens: usage?.prompt_tokens ?? null,
    generated_text: generatedText,
  }

  function consumeSseLine(line) {
    if (!line.startsWith('data:')) return
    const data = line.slice(5).trim()
    if (!data || data === '[DONE]') return
    const event = JSON.parse(data)
    if (event.usage) usage = event.usage
    const delta = event.choices?.[0]?.delta || {}
    const text = `${delta.reasoning_content || ''}${delta.content || ''}`
    if (!text) return
    if (firstContentAt === null) firstContentAt = performance.now()
    contentEvents += 1
    generatedText += text
  }
}

function summarize(measured) {
  const texts = new Set(measured.map((run) => run.generated_text))
  const decodeRates = measured
    .map((run) => run.decode_tokens_per_second)
    .filter((value) => value != null)
  return {
    mean_ttft_ms: mean(measured.map((run) => run.ttft_ms)),
    mean_total_ms: mean(measured.map((run) => run.total_ms)),
    mean_post_first_ms: mean(measured.map((run) => run.post_first_ms)),
    mean_decode_tokens_per_second: decodeRates.length ? mean(decodeRates) : null,
    min_total_ms: Math.min(...measured.map((run) => run.total_ms)),
    max_total_ms: Math.max(...measured.map((run) => run.total_ms)),
    identical_generated_text: texts.size === 1,
  }
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let index = 0; index < argv.length; index += 2) {
    if (!argv[index]?.startsWith('--') || argv[index + 1] === undefined) {
      throw new Error(`expected --name value arguments; got ${argv[index] || '<end>'}`)
    }
    parsed.set(argv[index].slice(2), argv[index + 1])
  }
  return parsed
}

function required(name) {
  const value = args.get(name)
  if (!value) throw new Error(`--${name} is required`)
  return value
}

function positiveInt(value, name) {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${name} must be positive`)
  return parsed
}
