#!/usr/bin/env node
import assert from 'node:assert/strict'
import {
  modelInfoUrl,
  resolveHfSource,
  sourceLockFromModelInfo,
  validateLockAgainstSelection,
  validateRequestedRevision,
} from './hf-qualification-source.mjs'

const revision = '1'.repeat(40)
const sha256 = 'a'.repeat(64)
const info = {
  sha: revision,
  cardData: { license: 'Apache-2.0' },
  gated: false,
  private: false,
  siblings: [{
    rfilename: 'weights/model-q8_0.gguf',
    size: 1234,
    lfs: { size: 1234, sha256 },
  }],
}

assert.equal(
  modelInfoUrl('org/model name', revision),
  `https://huggingface.co/api/models/org/model%20name/revision/${revision}?blobs=true`,
)
const lock = sourceLockFromModelInfo({
  repo: 'org/model',
  file: 'weights/model-q8_0.gguf',
  requestedRevision: revision,
  info,
})
assert.equal(lock.revision, revision)
assert.equal(lock.size_bytes, 1234)
assert.equal(lock.sha256, sha256)
assert.equal(lock.license, 'apache-2.0')
assert.equal(lock.download_url, `https://huggingface.co/org/model/resolve/${revision}/weights/model-q8_0.gguf?download=true`)

const namedOtherLicense = structuredClone(info)
namedOtherLicense.cardData = { license: 'other', license_name: 'lfm1.0' }
assert.equal(
  sourceLockFromModelInfo({
    repo: 'org/model',
    file: 'weights/model-q8_0.gguf',
    requestedRevision: revision,
    info: namedOtherLicense,
  }).license,
  'LFM Open License v1.0',
)

assert.throws(
  () => sourceLockFromModelInfo({ repo: 'org/model', file: 'missing.gguf', requestedRevision: revision, info }),
  /is absent/,
)
assert.throws(
  () => sourceLockFromModelInfo({ repo: 'org/model', file: 'weights/model-q8_0.gguf', requestedRevision: '2'.repeat(40), info }),
  /expected/,
)
const noHash = structuredClone(info)
delete noHash.siblings[0].lfs.sha256
assert.throws(
  () => sourceLockFromModelInfo({ repo: 'org/model', file: 'weights/model-q8_0.gguf', requestedRevision: revision, info: noHash }),
  /exact-byte qualification cannot proceed/,
)

let requestedUrl = null
const fetched = await resolveHfSource({
  repo: 'org/model',
  file: 'weights/model-q8_0.gguf',
  revision,
  fetchImpl: async (url) => {
    requestedUrl = url
    return { ok: true, json: async () => info }
  },
})
assert.equal(requestedUrl, modelInfoUrl('org/model', revision))
assert.equal(fetched.sha256, sha256)

const selected = {
  row_id: 'fixture_row',
  repo: 'org/model',
  file: 'weights/model-q8_0.gguf',
  revision,
  expected: { size_bytes: 1234, sha256, license: 'apache-2.0' },
}
assert.doesNotThrow(() => validateRequestedRevision(selected, revision))
assert.throws(() => validateRequestedRevision(selected, '2'.repeat(40)), /cannot override pinned roster revision/)
assert.doesNotThrow(() => validateLockAgainstSelection(lock, selected))
const wrongIdentity = structuredClone(lock)
wrongIdentity.sha256 = 'b'.repeat(64)
assert.throws(
  () => validateLockAgainstSelection(wrongIdentity, selected),
  /resolved source does not match roster row fixture_row: sha256/,
  'row-labelled locks must agree with the roster exact-byte identity',
)

console.log('test-hf-qualification-source: all checks passed')
