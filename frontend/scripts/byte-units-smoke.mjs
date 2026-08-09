#!/usr/bin/env node
/* The UI and the CLI must not disagree about how big a file is.
 *
 * `formatBytes` divided by 1024 while labelling the result KB/MB/GB, so every
 * size in the UI was a binary magnitude wearing an SI name. The same GGUF read
 * "3.4 GB" from `camelid pull` and "3.2 GB" on the Models page, and the second
 * number is what a user compares against the Hub's listing and their own disk.
 *
 * This pins the formatter to the same decimal convention `src/catalog.rs` uses
 * (`bytes as f64 / 1e9`, rendered `{:.1} GB`) and checks it against the real
 * catalog sizes rather than invented ones, so a future edit that reintroduces
 * 1024 fails here instead of shipping two numbers for one file.
 *
 * Pure module test -- no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { formatBytes } from '../src/lib/formatters.js'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))

/* What the Rust side prints for a GB-scale row: `{:.1} GB` over bytes / 1e9. */
function rustGb(bytes) {
  return `${(bytes / 1e9).toFixed(1)} GB`
}

/* Real catalog rows, so this cannot pass against a convenient fixture. */
const supportedModels = readFileSync(
  resolve(scriptDir, '../src/lib/supportedModels.js'),
  'utf8',
)
const sizes = [...supportedModels.matchAll(/size_bytes:\s*(\d+)/g)].map((m) => Number(m[1]))
assert.ok(sizes.length >= 2, `expected several catalog sizes, found ${sizes.length}`)

for (const bytes of sizes) {
  if (bytes < 1e9) continue // only GB-scale rows are directly comparable
  const ui = formatBytes(bytes)
  const cli = rustGb(bytes)
  assert.equal(
    ui,
    cli,
    `UI and CLI disagree for ${bytes} bytes: UI "${ui}" vs CLI "${cli}". ` +
      'formatBytes must divide by 1000 to match the SI labels it prints.',
  )
}

/* Exact boundaries, independent of the catalog contents. */
assert.equal(formatBytes(0), '0 B')
assert.equal(formatBytes(999), '999 B')
assert.equal(formatBytes(1000), '1.0 KB')
assert.equal(formatBytes(1_000_000), '1.0 MB')
assert.equal(formatBytes(1_000_000_000), '1.0 GB')
assert.equal(formatBytes(null), '—')
assert.equal(formatBytes(undefined), '—')

/* The regression itself: 1024-based maths would render this 3.2 GB. */
assert.equal(formatBytes(3_421_898_816), '3.4 GB')

console.log(`byte-units smoke OK (${sizes.length} catalog sizes checked)`)
