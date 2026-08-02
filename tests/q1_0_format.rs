//! Q1_0 (type id 41) decodes bit-exactly against the vendor's own
//! expansion of the same weights.
//!
//! Q1_0 is sign-only 1-bit quantization: a 128-element / 18-byte block
//! `{ f16 d; u8 qs[16] }` at 1.125 bpw, where element `j` is bit `j % 8` of byte
//! `j / 8` (LSB-first, sequential) and the value is `+d` when the bit is set and
//! `-d` when it is clear. The representable set is exactly `{-d, +d}`.
//!
//! WHY THIS FIXTURE IS NOT A SELF-CONSISTENCY CHECK: the expected values in
//! `tests/fixtures/dequant/q1_0_real_blocks.json` are not produced by Camelid. They
//! are read verbatim out of `prism-ml/Bonsai-1.7B-unpacked`'s `model.safetensors` —
//! PrismML's own f16 expansion of the very same 1-bit weights that the GGUF packs.
//! The wire bytes come verbatim from the GGUF. So this arbitrates Camelid's bit
//! order and sign convention against an independent vendor artifact.
//!
//! The discriminating property is element ORDER. An MSB-first decode of this same
//! tensor agrees with the vendor expansion on only ~50% of elements (chance, since
//! the signs are ~balanced) while still producing entirely plausible-looking
//! weights — which is why order is pinned on real data here and on a hand-built
//! block in `tensor::tests::q1_0_dequant_matches_the_reference_layout`.
//!
//! Every comparison is on `f32::to_bits()`, never float equality, so +0.0 and -0.0
//! stay distinguishable (the `neg_d = -d` seam in the decoder).

use std::path::{Path, PathBuf};

use camelid::tensor::{decode_q1_0_tensor, Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMENTS};
use serde_json::Value;

fn load_fixture() -> Value {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dequant")
        .join("q1_0_real_blocks.json");
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn hex_bytes(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex string");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex byte"))
        .collect()
}

/// The fixture must describe the format Camelid actually implements, or the
/// comparison below silently tests nothing.
#[test]
fn fixture_pins_the_block_geometry() {
    let doc = load_fixture();
    assert_eq!(
        doc["block_elements"].as_u64().unwrap() as usize,
        Q1_0_BLOCK_ELEMENTS
    );
    assert_eq!(
        doc["block_bytes"].as_u64().unwrap() as usize,
        Q1_0_BLOCK_BYTES
    );
    assert_eq!(
        doc["provenance"]["route"].as_str().unwrap(),
        "independent-vendor-expansion",
        "expected values must come from the vendor artifact, not from Camelid"
    );
}

#[test]
fn real_blocks_decode_bit_exactly_against_the_vendor_expansion() {
    let doc = load_fixture();
    let blocks = doc["blocks"].as_array().expect("blocks array");
    assert_eq!(
        blocks.len(),
        16,
        "fixture samples 16 blocks across the tensor"
    );

    for block in blocks {
        let b = block["b"].as_u64().unwrap();
        let wire = hex_bytes(block["wire_hex"].as_str().expect("wire_hex"));
        assert_eq!(wire.len(), Q1_0_BLOCK_BYTES, "block {b} wire length");

        let expected: Vec<f32> = block["expected"]
            .as_array()
            .expect("expected array")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect();
        assert_eq!(
            expected.len(),
            Q1_0_BLOCK_ELEMENTS,
            "block {b} expected length"
        );

        let got = decode_q1_0_tensor("fixture", &wire, Q1_0_BLOCK_ELEMENTS)
            .unwrap_or_else(|e| panic!("block {b} decode: {e}"));

        for (j, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "block {b} element {j}: decoded {g} (0x{:08X}) != vendor {e} (0x{:08X})",
                g.to_bits(),
                e.to_bits()
            );
        }

        // Sign-only: exactly one magnitude per block, and it is the block's own f16 d.
        let d = block["d"].as_f64().unwrap() as f32;
        assert!(
            got.iter().all(|v| v.abs().to_bits() == d.abs().to_bits()),
            "block {b} is not constant-magnitude — Q1_0 cannot represent that"
        );
    }
}

/// An MSB-first decode is the realistic way to get this wrong, and it still yields
/// plausible weights. Pin that the fixture actually rejects it, so a future decoder
/// regression cannot pass by luck.
#[test]
fn the_fixture_rejects_a_reversed_bit_order() {
    let doc = load_fixture();
    let blocks = doc["blocks"].as_array().unwrap();

    let mut agree = 0usize;
    let mut total = 0usize;
    for block in blocks {
        let wire = hex_bytes(block["wire_hex"].as_str().unwrap());
        let d = block["d"].as_f64().unwrap() as f32;
        let expected: Vec<f32> = block["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for j in 0..Q1_0_BLOCK_ELEMENTS {
            // Deliberately wrong: bit (7 - j % 8) instead of bit (j % 8).
            let bit = (wire[2 + j / 8] >> (7 - (j % 8))) & 1;
            let wrong = if bit == 1 { d } else { -d };
            if wrong.to_bits() == expected[j].to_bits() {
                agree += 1;
            }
            total += 1;
        }
    }

    assert!(
        agree < total,
        "a reversed bit order matched the vendor expansion everywhere — the fixture \
         does not discriminate element order"
    );
}
