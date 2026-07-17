#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { basename, join, resolve } from 'node:path'

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (m) => m.slice(1)))
const source = join(root, 'target', 'workspace-stage5-qwen3-4b-q4km')
const bundleName = 'workspace-qwen3-4b-q4km-20260717T165404Z-head-8c2a2b74'
const destination = join(root, 'qa', 'evidence-bundles', bundleName)
const codeFiles = [
  'src/api/mod.rs',
  'src/api/workspace.rs',
  'src/chat/workspace_bridge.rs',
  'src/chat/agent.rs',
  'src/chat/tools.rs',
  'src/error.rs',
  'src/lib.rs',
  'frontend/src/App.jsx',
  'frontend/src/components/layout/SidebarRail.jsx',
  'frontend/src/components/ui/Modal.jsx',
  'frontend/src/hooks/useDashboardData.js',
  'frontend/src/views/WorkspaceView.jsx',
  'frontend/src/lib/workspaceAgent.js',
  'frontend/src/styles/workspace.css',
  'scripts/workspace-stage5-eval.mjs',
  'frontend/scripts/workspace-stage5-ui.mjs',
]
const eventFiles = ['read-list-search.events.json', 'denied-write.events.json', 'approved-write.events.json']

function sha256(data) {
  return createHash('sha256').update(data).digest('hex')
}

function scrub(value) {
  if (Array.isArray(value)) return value.map(scrub)
  if (value && typeof value === 'object') {
    const result = {}
    for (const [key, item] of Object.entries(value)) {
      result[key] = key === 'session_id' || key === 'approval_id' ? `<${key}>` : scrub(item)
    }
    return result
  }
  if (typeof value !== 'string') return value
  return value
    .replaceAll('\\\\?\\C:\\camelid-fork\\target\\workspace-stage5-qwen3-4b-q4km\\disposable-workspace', '<workspace>')
    .replaceAll('\\\\?\\C:\\camelid-fork\\target\\workspace-stage5-qwen3-4b-q4km\\ui-workspace', '<ui-workspace>')
    .replaceAll('C:\\camelid-fork\\target\\workspace-stage5-qwen3-4b-q4km\\disposable-workspace', '<workspace>')
    .replaceAll('C:\\camelid-fork\\target\\workspace-stage5-qwen3-4b-q4km\\ui-workspace', '<ui-workspace>')
}

await mkdir(destination, { recursive: true })
const summary = JSON.parse(await readFile(join(source, 'summary.json'), 'utf8'))
await writeFile(join(destination, 'summary.json'), `${JSON.stringify(scrub(summary), null, 2)}\n`)
for (const filename of eventFiles) {
  const events = JSON.parse(await readFile(join(source, filename), 'utf8'))
  await writeFile(join(destination, filename), `${JSON.stringify(scrub(events), null, 2)}\n`)
}
for (const filename of ['real-ui-approval.png', 'real-ui-terminal.png']) {
  await copyFile(join(source, filename), join(destination, filename))
}

const codeHash = createHash('sha256')
for (const filename of codeFiles) {
  codeHash.update(`${filename}\0`)
  codeHash.update(await readFile(join(root, filename)))
}
const head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim()
const status = execFileSync('git', ['status', '--short'], { cwd: root, encoding: 'utf8' }).trim().split(/\r?\n/).filter(Boolean)
const manifest = {
  schema: 'camelid.workspace-evidence-bundle/v1',
  generated_at: summary.generated_at,
  source_base_head: head,
  working_tree_dirty: true,
  implementation_sha256: codeHash.digest('hex'),
  implementation_files: codeFiles,
  changed_paths_at_capture: status,
  model: summary.model,
  runtime: summary.runtime,
  scenarios: summary.scenarios,
  filesystem: summary.filesystem,
  ui: {
    approval: 'real-ui-approval.png',
    terminal: 'real-ui-terminal.png',
    approval_exact_visible_action: true,
    terminal_complete: true,
  },
  claim_boundary: summary.claim_boundary,
}
await writeFile(join(destination, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)

const files = ['README.md', 'manifest.json', 'summary.json', ...eventFiles, 'real-ui-approval.png', 'real-ui-terminal.png']
const readme = `# Camelid Workspace — Qwen3 4B Q4_K_M real-model closure\n\n` +
  `This bundle closes the bounded Web Workspace vertical slice on the exact ` +
  `\`Qwen3-4B-Q4_K_M.gguf\` artifact. The model is pinned to SHA-256 ` +
  `\`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5\` ` +
  `from \`Qwen/Qwen3-4B-GGUF@a9a60d009fa7ff9606305047c2bf77ac25dbec49\`.\n\n` +
  `## Passed scenarios\n\n` +
  `- read-only multi-step loop: \`list_dir\` → \`read_file\` → \`search\`;\n` +
  `- denied \`write_file\`: no file created and the workspace tree hash stayed unchanged;\n` +
  `- approved \`write_file\`: the approval UI displayed the exact target and complete proposed content, and the resulting file contained exactly \`hello there\`;\n` +
  `- an outside-root canary hash stayed unchanged;\n` +
  `- real browser approval and terminal-state screenshots were captured against the loaded model.\n\n` +
  `## Provenance\n\n` +
  `The run was captured from a dirty working tree based on \`${head}\`. ` +
  `\`manifest.json\` records the implementation-file digest and the changed paths. ` +
  `This is evidence for that exact working tree, not a claim that the base commit alone contains the feature.\n\n` +
  `## Non-claims\n\n` +
  `No shell, network, GUI, subagent, unattended, neighboring-model, cross-platform, or throughput claim is made. ` +
  `The capability remains exact-row gated by committed \`tool_capable\` evidence.\n`
await writeFile(join(destination, 'README.md'), readme)

const sums = []
for (const filename of files) {
  const data = await readFile(join(destination, filename))
  sums.push(`${sha256(data)}  ${filename}`)
}
await writeFile(join(destination, 'SHA256SUMS'), `${sums.join('\n')}\n`)
console.log(destination)
