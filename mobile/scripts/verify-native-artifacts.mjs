import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const moduleRoot = join(root, 'modules', 'camelid-remote-crypto', 'android');
const manifestPath = join(moduleRoot, 'native-artifacts.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

if (manifest.schema !== 'camelid.remote-native-artifacts/v1') {
  throw new Error('Unsupported native artifact manifest schema');
}

const files = [
  [manifest.kotlin_binding.library, manifest.kotlin_binding.sha256],
  ...manifest.artifacts.map((artifact) => [artifact.library, artifact.source_sha256]),
];

for (const [relativePath, expected] of files) {
  if (typeof relativePath !== 'string' || !/^[a-f0-9]{64}$/.test(expected)) {
    throw new Error('Invalid native artifact manifest entry');
  }
  const bytes = await readFile(join(moduleRoot, relativePath));
  const actual = createHash('sha256').update(bytes).digest('hex');
  if (actual !== expected) {
    throw new Error(`Native artifact hash mismatch: ${relativePath}`);
  }
}

console.log(`NATIVE_ARTIFACTS_VERIFIED=${files.length}`);
