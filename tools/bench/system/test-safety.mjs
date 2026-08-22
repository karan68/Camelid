#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { acquireCampaignLock, assertMinimumFreeDisk } from './lib/safety.mjs'

const temp = await mkdtemp(join(tmpdir(), 'camelid-benchmark-safety-'))
const lockPath = join(temp, 'host.lock')

try {
  const lock = await acquireCampaignLock(lockPath, {
    campaignId: 'safety-test',
    createdUtc: '2026-08-23T00:00:00Z',
  })
  await access(lockPath)
  await assert.rejects(
    () => acquireCampaignLock(lockPath, {
      campaignId: 'competing-test',
      createdUtc: '2026-08-23T00:00:00Z',
    }),
    /already locked/,
  )
  await lock.release()
  await assert.rejects(() => access(lockPath), { code: 'ENOENT' })
  await lock.release()

  const second = await acquireCampaignLock(lockPath, {
    campaignId: 'after-release',
    createdUtc: '2026-08-23T00:00:00Z',
  })
  await second.release()

  const observations = await assertMinimumFreeDisk([join(temp, 'future', 'output')], 100, {
    inspect: async () => 1000,
  })
  assert.equal(observations.length, 1)
  await assert.rejects(
    () => assertMinimumFreeDisk([temp], 1001, { inspect: async () => 1000 }),
    /insufficient free disk/,
  )
} finally {
  await rm(temp, { recursive: true, force: true })
}

console.log('benchmark Phase 1 host lock and disk preflight: PASS')
