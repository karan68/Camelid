import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const output = join(root, "crates/camelid-remote-crypto-ffi/bindings");
const config = join(root, "tools/remote-crypto-bindgen/uniffi-global.toml");
const cargo = resolveCargo();

run(cargo, ["build", "-p", "camelid-remote-crypto-ffi"]);

const library = platformLibrary();
if (!existsSync(library)) {
  throw new Error(`compiled UniFFI library is missing: ${library}`);
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
run(cargo, [
  "run",
  "-p",
  "camelid-remote-crypto-bindgen",
  "--",
  "generate",
  "--no-format",
  "--config",
  config,
  "--language",
  "swift",
  "--language",
  "kotlin",
  "--out-dir",
  output,
  library,
]);

const files = await walk(output);
const swift = await find(files, /CamelidRemoteCrypto\.swift$/);
const kotlin = await find(files, /camelid_remote_crypto_ffi\.kt$/);
const header = await find(files, /CamelidRemoteCryptoFFI\.h$/);
const modulemap = await find(files, /CamelidRemoteCryptoFFI\.modulemap$/);
const swiftText = await readFile(swift, "utf8");
const kotlinText = await readFile(kotlin, "utf8");

requireText(kotlinText, /^package ai\.camelid\.remote\.crypto$/m, "Kotlin package");
requireText(kotlinText, /android\.system\.SystemCleaner/, "Android cleaner");
requireText(swiftText, /canImport\(CamelidRemoteCryptoFFI\)/, "Swift FFI module");
for (const [label, text] of [
  ["Kotlin", kotlinText],
  ["Swift", swiftText],
]) {
  if (/StaticKeyMaterial/.test(text)) {
    throw new Error(`${label} exposes a secret-bearing key record`);
  }
  requireText(text, /GeneratedStaticKey/, `${label} opaque generated key`);
  requireText(text, /takePrivateKey/, `${label} one-shot private key method`);
}

const generated = [swift, header, modulemap, kotlin].sort();
const manifest = {
  schema: "camelid.remote-crypto-bindings/v1",
  generator: "uniffi/0.32.0",
  crate: "camelid-remote-crypto-ffi/0.0.0",
  files: [],
};
for (const path of generated) {
  const bytes = await readFile(path);
  manifest.files.push({
    path: relative(output, path).replaceAll("\\", "/"),
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  });
}
await writeFile(join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`remote crypto bindings: generated and sealed ${generated.length} files`);

function resolveCargo() {
  if (process.env.CARGO) return process.env.CARGO;
  if (process.platform === "win32" && process.env.USERPROFILE) {
    const candidate = join(process.env.USERPROFILE, ".cargo/bin/cargo.exe");
    if (existsSync(candidate)) return candidate;
  }
  return "cargo";
}

function platformLibrary() {
  const directory = join(root, "target/debug");
  if (process.platform === "win32") return join(directory, "camelid_remote_crypto_ffi.dll");
  if (process.platform === "darwin") return join(directory, "libcamelid_remote_crypto_ffi.dylib");
  return join(directory, "libcamelid_remote_crypto_ffi.so");
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} exited ${result.status}`);
  }
}

async function walk(directory) {
  const paths = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) paths.push(...(await walk(path)));
    else paths.push(path);
  }
  return paths;
}

async function find(paths, pattern) {
  const matches = paths.filter((path) => pattern.test(path));
  if (matches.length !== 1) {
    throw new Error(`expected one ${pattern}, found ${matches.length}`);
  }
  return matches[0];
}

function requireText(text, pattern, label) {
  if (!pattern.test(text)) throw new Error(`${label} was not generated`);
}