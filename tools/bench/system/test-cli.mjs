#!/usr/bin/env node
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { validatePlan } from './lib/contracts.mjs'
import { controllerManifest } from './lib/controller-manifest.mjs'
import { sha256Bytes } from './lib/digest.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const repoRoot = resolve(systemRoot, '../../..')
const cli = resolve(systemRoot, 'cli.mjs')
const temp = await mkdtemp(join(tmpdir(), 'camelid-benchmark-cli-'))

try {
  const base = await sourceRepo('base')
  const head = await sourceRepo('head')
  const modelPath = join(temp, 'synthetic.gguf')
  const modelBytes = Buffer.from('synthetic model bytes')
  await writeFile(modelPath, modelBytes)
  const promptPath = join(temp, 'prompt.txt')
  await writeFile(promptPath, 'Explain exact benchmark provenance.\n', 'utf8')
  const controller = await controllerManifest(systemRoot)
  const campaign = {
    schema: 'camelid.benchmark.campaign/v1',
    campaign_id: 'cli-plan-test',
    mode: 'ablation',
    created_utc: '2026-08-23T00:00:00Z',
    controller: {
      source_manifest_sha256: controller.sha256,
      version: 'phase1-cli-test',
    },
    repository_root: repoRoot,
    source_arms: [arm('base', base), arm('head', head)],
    models: [{
      id: 'synthetic-row',
      artifact_path: modelPath,
      expected_sha256: sha256Bytes(modelBytes),
      quantization: 'synthetic',
    }],
    workloads: [{
      id: 'rt_decode_short',
      adapter: 'runtime-camelid',
      model_id: 'synthetic-row',
      prompt_file: promptPath,
      max_tokens: 4,
      warmup: true,
      deterministic: true,
      threads: 1,
      backend: {
        requested: 'cpu_deterministic',
        assertion: 'deterministic_no_offload',
      },
      primary_metrics: ['tokens_per_second'],
      schedule: 'balanced_rotation',
      repetitions: 2,
      timeout_ms: 1000,
    }],
    resources: {
      minimum_free_disk_bytes: 1,
    },
    security: {
      network: 'deny',
      trust_tier: 'local_exploratory',
    },
  }
  const configPath = join(temp, 'campaign.json')
  const planPath = join(temp, 'plan.json')
  await writeFile(configPath, `${JSON.stringify(campaign, null, 2)}\n`, 'utf8')

  const planned = run(['plan', '--config', configPath, '--out', planPath])
  assert.equal(planned.status, 0, planned.stderr)
  const stdoutPlan = JSON.parse(planned.stdout)
  const filePlan = JSON.parse(await readFile(planPath, 'utf8'))
  assert.deepEqual(stdoutPlan, filePlan)
  validatePlan(filePlan)
  assert.deepEqual(filePlan.workloads[0].order, ['base', 'head', 'head', 'base'])
  assert.equal(filePlan.models[0].artifact_sha256, sha256Bytes(modelBytes))

  const digest = run(['digest'])
  assert.equal(digest.status, 0, digest.stderr)
  assert.equal(digest.stdout.trim(), controller.sha256)

  const wrongController = structuredClone(campaign)
  wrongController.controller.source_manifest_sha256 = '0'.repeat(64)
  const wrongPath = join(temp, 'wrong-controller.json')
  await writeFile(wrongPath, `${JSON.stringify(wrongController)}\n`, 'utf8')
  const refused = run(['plan', '--config', wrongPath])
  assert.equal(refused.status, 1)
  assert.match(refused.stderr, /controller digest is/)
} finally {
  await rm(temp, { recursive: true, force: true })
}

console.log('benchmark Phase 1 CLI plan integration: PASS')

function arm(id, source) {
  return {
    id,
    source_dir: source.path,
    expected_git_sha: source.sha,
    cargo_path: process.execPath,
    toolchain: '1.89.0',
    features: [],
    target_dir: join(temp, 'target', id),
  }
}

async function sourceRepo(name) {
  const path = join(temp, name)
  git(['init', path], temp)
  git(['config', 'user.name', 'Benchmark Test'], path)
  git(['config', 'user.email', 'benchmark@example.invalid'], path)
  await writeFile(join(path, 'Cargo.lock'), `# ${name}\n`, 'utf8')
  git(['add', 'Cargo.lock'], path)
  git(['commit', '-m', `fixture ${name}`], path)
  return { path, sha: git(['rev-parse', 'HEAD'], path).stdout.trim() }
}

function git(args, cwd) {
  const result = spawnSync('git', args, { cwd, encoding: 'utf8', windowsHide: true })
  assert.equal(result.status, 0, result.stderr || result.stdout)
  return result
}

function run(args) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
    windowsHide: true,
  })
}
