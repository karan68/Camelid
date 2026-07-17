// BASALT Amendment 3 §1.2 — synthetic NVFP4 refusal-trip fixtures (WIRE lane).
//
// Provenance: authored for the Amendment 3 closure commit (2026-07-16), Phase 3
// branch basalt/phase3-cpu-eval. Byte layout follows the GGUF v3 reader in
// src/gguf/reader.rs (magic/version/counts, string = u64 len + utf8, tensor
// descriptor = name/n_dims/dims i64/type i32/offset u64, data section aligned
// to general.alignment default 32). NVFP4 = pin type id 40, 64-element/36-byte
// superblock {d[4] UE4M3, qs[32]} (D-B1). Output is fully deterministic: no
// timestamps, no randomness — re-running MUST reproduce byte-identical files
// (shas pinned in tests/nvfp4_wire_lane_refusals.rs and SHA256SUMS alongside).
//
// Two fixtures, both real parseable GGUF v3 files < 4 KB:
//   nvfp4_sidecar_trip.gguf   — one NVFP4 tensor + `.scale` + `.input_scale`
//       sidecar tensors: trips the D-B2 sidecar refusal END-TO-END through
//       Gemma4Runtime::load (the check fires right after read_metadata, before
//       any config parsing, so no full gemma4 metadata is needed).
//   nvfp4_nan_sentinel_trip.gguf — one NVFP4 tensor whose wire bytes carry a
//       raw 0x7F UE4M3 scale byte (D17/T5 NaN sentinel) at the correct data
//       offset. Full-file load CANNOT reach the WireQuant sentinel scan (config
//       parsing precedes WireQuant construction and fails first on missing
//       gemma4 keys) — the integration test drives the scan seam directly on
//       these bytes and documents that ordering (honest cells only).
//
// Usage: node scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs
// Writes tests/fixtures/gguf/{nvfp4_sidecar_trip,nvfp4_nan_sentinel_trip}.gguf
// plus tests/fixtures/gguf/SHA256SUMS (LF line endings — CRLF breaks verifiers).

import { mkdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

const OUT_DIR = join("tests", "fixtures", "gguf");
const ALIGNMENT = 32; // reader default (no general.alignment KV written)
const GGUF_VERSION = 3;
const T_F32 = 0;
const T_NVFP4 = 40; // pin GGML_TYPE_NVFP4 (D-B1)

// --- little-endian emit helpers over a growable byte array -----------------
class W {
  constructor() {
    this.chunks = [];
    this.len = 0;
  }
  push(buf) {
    this.chunks.push(buf);
    this.len += buf.length;
  }
  u32(v) {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(v);
    this.push(b);
  }
  i32(v) {
    const b = Buffer.alloc(4);
    b.writeInt32LE(v);
    this.push(b);
  }
  u64(v) {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(v));
    this.push(b);
  }
  str(s) {
    const bytes = Buffer.from(s, "utf8");
    this.u64(bytes.length);
    this.push(bytes);
  }
  bytes(arr) {
    this.push(Buffer.from(arr));
  }
  alignTo(a) {
    const pad = (a - (this.len % a)) % a;
    if (pad > 0) this.push(Buffer.alloc(pad));
  }
  out() {
    return Buffer.concat(this.chunks, this.len);
  }
}

// One deterministic 36-byte NVFP4 superblock: 4 UE4M3 sub-scales + 32 qs bytes.
// Safe scales avoid the 0x7F/0xFF sentinels; the NaN variant plants 0x7F at d[0].
function nvfp4Block({ nanSentinel }) {
  const d = nanSentinel ? [0x7f, 0x40, 0x51, 0x66] : [0x38, 0x40, 0x51, 0x66];
  const qs = Array.from({ length: 32 }, (_, j) => (j * 7 + 3) % 256);
  return [...d, ...qs];
}

const f32le = (v) => {
  const b = Buffer.alloc(4);
  b.writeFloatLE(v);
  return b;
};

// tensors: [{ name, dims, type, data: Buffer }]
function buildGguf(tensors) {
  const w = new W();
  w.push(Buffer.from("GGUF", "ascii"));
  w.u32(GGUF_VERSION);
  w.u64(tensors.length); // tensor_count (i64)
  w.u64(1); // metadata_count (i64)

  // Single honest KV: this is a gemma4-lane fixture (the refusal under test
  // does not depend on it — the sidecar check runs before config parsing).
  w.str("general.architecture");
  w.i32(8); // string
  w.str("gemma4");

  // Tensor table with reader-contiguous relative offsets (each tensor's
  // rel_offset = align(prev_end, ALIGNMENT)).
  let rel = 0;
  const placed = [];
  for (const t of tensors) {
    w.str(t.name);
    w.u32(t.dims.length);
    for (const dim of t.dims) w.u64(dim);
    w.i32(t.type);
    w.u64(rel);
    placed.push({ ...t, rel });
    rel = Math.ceil((rel + t.data.length) / ALIGNMENT) * ALIGNMENT;
  }

  // Data section starts at align(header_end, ALIGNMENT). Each tensor's rel
  // offset is ALIGNMENT-aligned and data_start is too, so aligning the absolute
  // position before each push lands every tensor exactly at data_start + rel.
  w.alignTo(ALIGNMENT);
  for (const t of placed) {
    w.alignTo(ALIGNMENT);
    w.push(t.data);
  }
  return w.out();
}

const fixtures = [
  {
    file: "nvfp4_sidecar_trip.gguf",
    tensors: [
      {
        name: "blk.0.ffn_down.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: false })),
      },
      // ModelOpt-convention sidecar pair — the D-B2 trip wires.
      { name: "blk.0.ffn_down.weight.scale", dims: [1], type: T_F32, data: f32le(1.0) },
      { name: "blk.0.ffn_down.weight.input_scale", dims: [1], type: T_F32, data: f32le(1.0) },
    ],
  },
  {
    file: "nvfp4_nan_sentinel_trip.gguf",
    tensors: [
      {
        name: "blk.0.ffn_up.weight",
        dims: [64],
        type: T_NVFP4,
        data: Buffer.from(nvfp4Block({ nanSentinel: true })), // d[0] = 0x7F
      },
    ],
  },
];

mkdirSync(OUT_DIR, { recursive: true });
const sums = [];
for (const f of fixtures) {
  const bytes = buildGguf(f.tensors);
  if (bytes.length >= 4096) throw new Error(`${f.file}: fixture must stay tiny (<4 KB)`);
  writeFileSync(join(OUT_DIR, f.file), bytes);
  const sha = createHash("sha256").update(bytes).digest("hex");
  sums.push(`${sha}  ${f.file}`);
  console.log(`${f.file}: ${bytes.length} B sha256=${sha}`);
}
writeFileSync(join(OUT_DIR, "SHA256SUMS"), sums.join("\n") + "\n"); // LF only
console.log("wrote", join(OUT_DIR, "SHA256SUMS"));
