#!/usr/bin/env node
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { writeBenchmarkBundle, verifyBundleChecksums } from './bundle.mjs'
import { controllerManifest } from './lib/controller-manifest.mjs'
import { canonicalJson } from './lib/digest.mjs'
import { acquireCampaignLock, assertMinimumFreeDisk } from './lib/safety.mjs'
import { resolveCampaignPlan, serializePlan } from './planner.mjs'
import { prepareArms } from './prepare.mjs'
import { runRuntimeCampaign } from './adapters/runtime-camelid.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const args = parseArgs(process.argv.slice(2))
const command = args.positionals[0]
let activeRun = null

try {
  if (command === 'digest') {
    const controller = await controllerManifest(systemRoot)
    console.log(controller.sha256)
  } else if (command === 'plan') {
    const { plan } = await loadAndResolve(args)
    const text = serializePlan(plan)
    if (args.values.has('out')) await writeText(resolve(args.values.get('out')), text)
    process.stdout.write(text)
  } else if (command === 'run') {
    const { request, plan } = await loadAndResolve(args)
    const outRoot = resolve(args.values.get('out-root') ?? joinDefaultOutput(plan.repository_root))
    const outputDir = resolve(outRoot, plan.campaign_id)
    await mkdir(outRoot, { recursive: true })
    await assertMinimumFreeDisk(
      [outRoot, ...plan.source_arms.map((arm) => arm.build.target_dir)],
      plan.resources.minimum_free_disk_bytes,
    )
    const lock = await acquireCampaignLock(resolve(outRoot, '.camelid-benchmark.lock'), {
      campaignId: plan.campaign_id,
      createdUtc: plan.created_utc,
    })
    try {
      await requireNewDirectory(outputDir)
      await mkdir(outputDir, { recursive: true })
      activeRun = { outputDir, campaignId: plan.campaign_id }
      await writeText(resolve(outputDir, 'plan.json'), serializePlan(plan))

      let preparedArms
      let preparationMode
      if (args.values.has('prepared')) {
        if (request.mode !== 'ablation' || request.security.trust_tier !== 'local_exploratory') {
          throw new Error('--prepared is restricted to local_exploratory ablation campaigns')
        }
        preparedArms = JSON.parse(await readFile(resolve(args.values.get('prepared')), 'utf8'))
        preparationMode = 'supplied_local_ablation'
      } else {
        preparedArms = await prepareArms(plan, {
          logDir: resolve(outputDir, 'build'),
        })
        preparationMode = 'built_from_plan'
      }

      const runtime = await runRuntimeCampaign(plan, preparedArms, { outputDir })
      const bundle = await writeBenchmarkBundle({
        plan,
        preparedArms,
        samples: runtime.samples,
        executions: runtime.executions,
        outputDir,
        preparationMode,
      })
      const verification = await verifyBundleChecksums(outputDir)
      if (!verification.ok) throw new Error(`bundle checksum verification failed: ${verification.failures.join('; ')}`)
      console.log(`bundle_dir=${outputDir}`)
      console.log(`state=${bundle.manifest.state}`)
      activeRun = null
      if (bundle.manifest.state !== 'COMPLETE_VALID') process.exitCode = 1
    } finally {
      await lock.release()
    }
  } else {
    process.stdout.write(usage())
    process.exitCode = command ? 2 : 0
  }
} catch (error) {
  if (activeRun) {
    await writeText(resolve(activeRun.outputDir, 'failure.json'), canonicalJson({
      schema: 'camelid.benchmark.failure/v1',
      campaign_id: activeRun.campaignId,
      state: 'INCOMPLETE',
      error_type: error.name,
      error_message: error.message,
    })).catch(() => {})
  }
  console.error(`benchmark Phase 1 ${command ?? 'command'} failed: ${error.message}`)
  process.exitCode = 1
}

async function loadAndResolve(parsed) {
  const configPath = parsed.values.get('config')
  if (!configPath) throw new Error('--config is required')
  const absoluteConfig = resolve(configPath)
  const base = dirname(absoluteConfig)
  const request = JSON.parse(await readFile(absoluteConfig, 'utf8'))
  resolvePaths(request, base)
  const controller = await controllerManifest(systemRoot)
  if (request.controller?.source_manifest_sha256 !== controller.sha256) {
    throw new Error(`controller digest is ${controller.sha256}, config pins ${request.controller?.source_manifest_sha256 ?? 'nothing'}`)
  }
  const plan = await resolveCampaignPlan(request)
  return { request, plan }
}

function resolvePaths(request, base) {
  request.repository_root = localPath(base, request.repository_root)
  for (const arm of request.source_arms ?? []) {
    arm.source_dir = localPath(base, arm.source_dir)
    arm.cargo_path = localPath(base, arm.cargo_path)
    arm.target_dir = localPath(base, arm.target_dir)
  }
  for (const model of request.models ?? []) model.artifact_path = localPath(base, model.artifact_path)
  for (const workload of request.workloads ?? []) workload.prompt_file = localPath(base, workload.prompt_file)
}

function localPath(base, value) {
  return isAbsolute(value) ? value : resolve(base, value)
}

function parseArgs(argv) {
  const positionals = []
  const values = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (!token.startsWith('--')) {
      positionals.push(token)
      continue
    }
    const [name, inline] = token.slice(2).split('=', 2)
    if (inline !== undefined) {
      values.set(name, inline)
      continue
    }
    const next = argv[index + 1]
    if (!next || next.startsWith('--')) throw new Error(`--${name} requires a value`)
    values.set(name, next)
    index += 1
  }
  return { positionals, values }
}

async function requireNewDirectory(path) {
  try {
    const info = await stat(path)
    if (!info.isDirectory()) throw new Error(`output path exists and is not a directory: ${path}`)
    const entries = await readdir(path)
    if (entries.length > 0) throw new Error(`output directory already contains files: ${path}`)
  } catch (error) {
    if (error.code === 'ENOENT') return
    throw error
  }
}

async function writeText(path, text) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, text, 'utf8')
}

function joinDefaultOutput(repositoryRoot) {
  return resolve(repositoryRoot, 'target', 'benchmark-runs')
}

function usage() {
  return `Camelid Phase 1 benchmark system\n\n` +
    `  node tools/bench/system/cli.mjs digest\n` +
    `  node tools/bench/system/cli.mjs plan --config <campaign.json> [--out <plan.json>]\n` +
    `  node tools/bench/system/cli.mjs run --config <campaign.json> [--out-root <dir>] [--prepared <prepared-arms.json>]\n`
}
