import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'

export function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

export async function sha256File(path) {
  const hash = createHash('sha256')
  await new Promise((resolve, reject) => {
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.once('error', reject)
    stream.once('end', resolve)
  })
  return hash.digest('hex')
}

export function canonicalJson(value) {
  return `${JSON.stringify(canonicalize(value), null, 2)}\n`
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value !== null && typeof value === 'object') {
    const result = {}
    for (const key of Object.keys(value).sort()) {
      const child = value[key]
      if (child === undefined) throw new TypeError(`cannot canonicalize undefined field ${key}`)
      result[key] = canonicalize(child)
    }
    return result
  }
  if (typeof value === 'number' && !Number.isFinite(value)) {
    throw new TypeError('cannot canonicalize a non-finite number')
  }
  return value
}
