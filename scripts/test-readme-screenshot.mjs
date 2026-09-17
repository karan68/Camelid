#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const repoRoot = new URL('..', import.meta.url)
const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8')
// The hero screenshot the README actually ships. Refreshed with the README
// itself (docs/assets/readme/, captured from the frontend at v0.7.7); this guard
// pins the bytes so the surface cannot change without an explicit update here.
const expectedAsset = 'docs/assets/readme/desktop-chat.png'
const retiredLightAsset = 'docs/assets/ui-screenshot-v2.png'
const expectedSha256 = 'e064e5e7eda8711cddbf54a538ce48e7cf32bb7eab78db4701e5ce440cec6b8d'
const expectedAlt = 'Camelid desktop chat with a pinned conversation, Markdown table, project context, and model selector'

assert.match(
  readme,
  new RegExp(`!\\[${expectedAlt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\]\\(${expectedAsset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`),
  'README must use the approved Camelid desktop chat screenshot',
)
assert.doesNotMatch(
  readme,
  new RegExp(retiredLightAsset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
  'README must not point at the retired light WebUI screenshot',
)
assert.match(
  readme,
  /Screenshots show interface features, not model-quality or performance results/i,
  'README caption must keep saying the screenshots are interface-only, not results',
)

const assetBytes = readFileSync(join(fileURLToPath(repoRoot), expectedAsset))
const actualSha256 = createHash('sha256').update(assetBytes).digest('hex')
assert.equal(
  actualSha256,
  expectedSha256,
  'approved README screenshot bytes changed; update this guard only with explicit product approval',
)

console.log('README screenshot guard passed')
