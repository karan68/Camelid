import { open, readFile, rm, stat, statfs } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

import { canonicalJson } from './digest.mjs'

export async function acquireCampaignLock(path, details) {
  const absolute = resolve(path)
  let handle
  try {
    handle = await open(absolute, 'wx')
  } catch (error) {
    if (error.code !== 'EEXIST') throw error
    let existing = 'unreadable lock'
    try {
      existing = (await readFile(absolute, 'utf8')).trim()
    } catch {}
    throw new Error(`benchmark host is already locked at ${absolute}: ${existing}`)
  }
  try {
    await handle.writeFile(canonicalJson({
      schema: 'camelid.benchmark.host-lock/v1',
      pid: process.pid,
      campaign_id: details.campaignId,
      created_utc: details.createdUtc,
    }), 'utf8')
    await handle.sync()
  } catch (error) {
    await handle.close().catch(() => {})
    await rm(absolute, { force: true }).catch(() => {})
    throw error
  }
  let released = false
  return {
    path: absolute,
    async release() {
      if (released) return
      released = true
      await handle.close()
      await rm(absolute, { force: true })
    },
  }
}

export async function assertMinimumFreeDisk(paths, minimumBytes, options = {}) {
  if (!Array.isArray(paths) || paths.length === 0) throw new RangeError('disk preflight requires at least one path')
  if (!Number.isSafeInteger(minimumBytes) || minimumBytes < 1) {
    throw new RangeError('minimum free disk bytes must be a positive safe integer')
  }
  const inspect = options.inspect ?? availableBytes
  const observations = []
  for (const path of [...new Set(paths.map((candidate) => resolve(candidate)))]) {
    const available = await inspect(path)
    observations.push({ path, available_bytes: available })
    if (!Number.isSafeInteger(available) || available < minimumBytes) {
      throw new Error(`insufficient free disk at ${path}: ${available} bytes available, ${minimumBytes} required`)
    }
  }
  return observations
}

async function availableBytes(path) {
  const existing = await nearestExisting(path)
  const info = await statfs(existing, { bigint: true })
  const available = info.bavail * info.bsize
  if (available > BigInt(Number.MAX_SAFE_INTEGER)) return Number.MAX_SAFE_INTEGER
  return Number(available)
}

async function nearestExisting(path) {
  let current = resolve(path)
  while (true) {
    try {
      const info = await stat(current)
      if (info.isDirectory()) return current
      current = dirname(current)
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
      const parent = dirname(current)
      if (parent === current) throw new Error(`no existing parent for disk preflight path ${path}`)
      current = parent
    }
  }
}
