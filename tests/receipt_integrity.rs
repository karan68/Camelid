//! CI gate for the committed sealed-receipt evidence base.
//!
//! This runs inside the standard `cargo test --all-targets` on every OS leg, so
//! a corrupted or hand-edited committed receipt (parity / distributed / agent)
//! fails the build without any workflow-file wiring. Pre-existing broken seals
//! are grandfathered by the in-code baseline in `camelid::receipt::audit`; a NEW
//! or further-edited corruption is a hard failure here.
//!
//! Run the same check by hand with `camelid verify-receipts qa`.

use std::path::Path;

#[test]
fn committed_sealed_receipts_are_intact_or_baselined() {
    let qa = Path::new(env!("CARGO_MANIFEST_DIR")).join("qa");
    assert!(
        qa.is_dir(),
        "qa/ not found at {} — the receipt-integrity gate needs the repo checkout",
        qa.display()
    );

    let report = camelid::receipt::audit::audit_dir(&qa).expect("audit qa/");

    assert!(
        report.ok(),
        "committed sealed receipts are corrupted and NOT grandfathered:\n{:#?}\n\
         Fix the receipt (or re-seal it); only add a documented \
         camelid::receipt::audit::BASELINE entry for genuine pre-existing debt.",
        report.mismatches
    );

    // Sanity: the gate must actually be exercising sealed receipts, not silently
    // scanning nothing (e.g. a moved qa/ or a broken walk).
    assert!(
        report.verified + report.allowlisted.len() > 0,
        "audit found no sealed receipts under qa/ — the gate would be inert"
    );
}
