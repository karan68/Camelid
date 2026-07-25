import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

const toolDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(toolDir, "../..");
const fixtures = join(root, "tests/fixtures/remote/v1");
const schemaNames = [
  "inner-message",
  "command",
  "event",
  "pairing-qr",
  "approval-record",
  "relay-envelope",
];

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function canonicalJson(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) {
      throw new Error("canonical authority JSON permits only safe integers");
    }
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  throw new Error(`unsupported canonical JSON value: ${typeof value}`);
}

function requireValid(validate, value, label) {
  if (!validate(value)) {
    throw new Error(`${label} should be valid: ${JSON.stringify(validate.errors)}`);
  }
}

function requireInvalid(validate, value, label) {
  if (validate(value)) {
    throw new Error(`${label} should fail closed`);
  }
}

const ajv = new Ajv2020({ allErrors: true, strict: true });
addFormats(ajv);

const schemas = new Map();
for (const name of schemaNames) {
  const schema = await readJson(join(fixtures, "schema", `${name}.schema.json`));
  schemas.set(name, schema);
  ajv.addSchema(schema);
}

const validate = Object.fromEntries(
  schemaNames.map((name) => [name, ajv.getSchema(schemas.get(name).$id)]),
);
for (const [name, validator] of Object.entries(validate)) {
  if (typeof validator !== "function") {
    throw new Error(`schema did not compile: ${name}`);
  }
}

const startMessage = await readJson(join(fixtures, "valid/start_turn_message.json"));
const pairingQr = await readJson(join(fixtures, "valid/pairing_qr.json"));
const approval = await readJson(join(fixtures, "valid/edit_file_approval_record.json"));
const event = await readJson(join(fixtures, "valid/event.json"));
const relayEnvelope = await readJson(join(fixtures, "valid/relay_envelope.json"));

requireValid(validate["inner-message"], startMessage, "inner message");
requireValid(validate.command, startMessage.payload, "start_turn command");
requireValid(validate["pairing-qr"], pairingQr, "pairing QR");
requireValid(validate["approval-record"], approval, "approval record");
requireValid(validate.event, event, "event");
requireValid(validate["relay-envelope"], relayEnvelope, "relay envelope");

const unknownCommand = await readJson(
  join(fixtures, "invalid/start_turn_unknown_field.json"),
);
const insecureQr = await readJson(join(fixtures, "invalid/pairing_qr_http.json"));
const mismatchedApproval = await readJson(
  join(fixtures, "invalid/approval_mismatched_tool.json"),
);
requireInvalid(validate.command, unknownCommand.payload, "unknown command authority");
requireInvalid(validate["pairing-qr"], insecureQr, "insecure pairing QR");
requireInvalid(
  validate["approval-record"],
  mismatchedApproval,
  "mismatched approval authority",
);

const manifest = await readJson(join(fixtures, "manifest.json"));
if (manifest.schema_count !== schemaNames.length) {
  throw new Error("manifest schema_count drifted");
}
if (
  manifest.max_chunk_data_bytes !==
    manifest.max_noise_record_bytes - manifest.noise_tag_bytes - manifest.chunk_header_bytes ||
  manifest.max_inner_message_bytes >
    manifest.max_chunk_data_bytes * manifest.max_message_chunks
) {
  throw new Error("manifest Noise/chunk bounds are inconsistent");
}
const canonical = canonicalJson(approval);
const digest = `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
if (digest !== manifest.approval_fixture_digest) {
  throw new Error(`approval digest drifted: ${digest}`);
}

console.log(
  `remote protocol fixtures: ${schemaNames.length} schemas, 6 valid checks, ` +
    `3 invalid checks, digest ${digest}`,
);