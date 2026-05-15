#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

const script = 'scripts/bench-llama3-same-host.mjs'
const tmp = await mkdtemp(join(tmpdir(), 'camelid-bench-plan-'))

try {
  const planPath = join(tmp, 'plan.json')
  const planRun = spawnSync(process.execPath, [
    script,
    '--print-plan',
    '--model', '/tmp/Camelid Test/Llama-3.2-1B-Instruct-Q8_0.gguf',
    '--model-id', 'llama32-1b-q8-plan',
    '--row-id', 'llama32_1b_instruct_q8_0',
    '--max-tokens', '8',
    '--warmup', '0',
    '--repeats', '2',
    '--threads', '4',
    '--out', planPath,
  ], { encoding: 'utf8' })

  assert.equal(planRun.status, 0, planRun.stderr)
  assert.match(planRun.stdout, /harness_command=node scripts\/bench-llama3-same-host\.mjs/)
  assert.match(planRun.stdout, /claim_boundary=.*1B.*Mixtral.*separate row-specific evidence/s)

  const plan = JSON.parse(await readFile(planPath, 'utf8'))
  assert.equal(plan.schema, 'camelid.same_host_llama3_benchmark_plan.v1')
  assert.equal(plan.model.row_id, 'llama32_1b_instruct_q8_0')
  assert.equal(plan.method.max_tokens, 8)
  assert.equal(plan.method.warmup, 0)
  assert.equal(plan.method.repeats, 2)
  assert.equal(plan.method.threads, 4)
  assert.equal(plan.method.expected_marker, 'CMLD-BENCH')
  assert.equal(plan.method.require_marker, false)
  assert.equal(plan.method.evidence_context.model_artifact.sha256, 'not_computed_in_plan_mode')
  assert.ok(plan.method.evidence_context.host_class.cpu_count >= 1)
  assert.equal(plan.method.resource_snapshots.pre_start.label, 'pre_start')
  assert.match(plan.commands.harness, /--row-id llama32_1b_instruct_q8_0/)
  assert.match(plan.commands.harness, /'\/tmp\/Camelid Test\/Llama-3\.2-1B-Instruct-Q8_0\.gguf'/)
  assert.match(plan.commands.llama_server, /--no-warmup/)
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('not tokenizer-ground-truth tokens')))
  assert.ok(plan.method.bounded_metrics.some((metric) => metric.includes('marker_presence')))
  assert.match(plan.outputs.guardrail, /--require-marker/)
  assert.match(plan.claim_boundary, /does not widen support/)
  assert.match(plan.claim_boundary, /production-throughput/)
  assert.match(plan.claim_boundary, /Mixtral claims/)

  const helpRun = spawnSync(process.execPath, [script, '--help'], { encoding: 'utf8' })
  assert.equal(helpRun.status, 0, helpRun.stderr)
  assert.match(helpRun.stdout, /--print-plan/)
  assert.match(helpRun.stdout, /JSON report schema: camelid\.same_host_llama3_benchmark\.v1/)
  assert.match(helpRun.stdout, /does not promote production throughput/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}
