#!/usr/bin/env node
import { spawn } from 'node:child_process'
import net from 'node:net'
import { fileURLToPath } from 'node:url'

const mode = process.argv[2]
const selfPath = fileURLToPath(import.meta.url)

switch (mode) {
  case 'normal':
    process.stdout.write('normal stdout\n')
    process.stderr.write('normal stderr\n')
    break
  case 'fail':
    process.stderr.write('intentional failure\n')
    process.exitCode = 7
    break
  case 'large':
    process.stdout.write('x'.repeat(128 * 1024))
    break
  case 'hang':
    setInterval(() => {}, 60_000)
    break
  case 'descendant': {
    spawn(process.execPath, [selfPath, 'listener'], {
      stdio: ['ignore', 'inherit', 'inherit'],
      windowsHide: true,
    })
    setInterval(() => {}, 60_000)
    break
  }
  case 'listener': {
    const server = net.createServer(() => {})
    server.listen(0, '127.0.0.1', () => {
      process.stdout.write(`BENCH_CHILD_PORT=${server.address().port}\n`)
    })
    break
  }
  default:
    process.stderr.write(`unknown mode ${mode}\n`)
    process.exitCode = 2
}
