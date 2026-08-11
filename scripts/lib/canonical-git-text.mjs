import { createHash } from 'node:crypto'

export function canonicalGitTextBytes(value) {
  if (!Buffer.isBuffer(value)) throw new TypeError('canonical Git text must be a Buffer')
  return Buffer.from(value.toString('utf8').replace(/\r\n/g, '\n'))
}

export function canonicalGitTextSha256(value) {
  return createHash('sha256').update(canonicalGitTextBytes(value)).digest('hex')
}
