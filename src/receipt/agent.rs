//! Standalone verifier for the sealed agent-family receipts, exposed through
//! `camelid verify-receipt`.
//!
//! The agent gates each emit a **tamper-evident** receipt sealed with the shared
//! [`receipt_id`](super::receipt_id_over) digest:
//!
//! - `agent-syscap-eval` → `camelid.agent-syscap-receipt/v1`
//! - `agent-orchestration-eval` → `camelid.agent-orchestration-receipt/v1`
//! - `agent-orchestration-bench` → `camelid.agent-orchestration-bench/v1`
//!
//! Unlike a parity receipt there is no model to re-run: verification is a
//! self-contained **tamper-evidence + honest-scope** check.
//!
//! 1. *Tamper-evidence* — the document's canonical body (every field except
//!    `receipt_id`, recursively key-sorted and compact) must hash to the stored
//!    `receipt_id`. Any added, removed, or altered field breaks the match.
//! 2. *Honest-scope* — a well-formed but over-claiming receipt is rejected. These
//!    gates promote no capability and (for orchestration) claim no speedup; a
//!    receipt whose own scope fields say otherwise is not a valid artifact of the
//!    gate that supposedly produced it.
//!
//! A verified agent receipt attests that the document is intact and claims
//! nothing it should not. Like every receipt in this crate, it promotes no
//! support claim.

use std::path::Path;

use serde_json::{Map, Value};

use super::receipt_id_over;

/// Schema stamped into an `agent-syscap-eval` receipt.
pub const SYSCAP_RECEIPT_SCHEMA_V1: &str = "camelid.agent-syscap-receipt/v1";
/// Schema stamped into an `agent-orchestration-eval` receipt.
pub const ORCHESTRATION_RECEIPT_SCHEMA_V1: &str = "camelid.agent-orchestration-receipt/v1";
/// Schema stamped into an `agent-orchestration-bench` receipt.
pub const ORCHESTRATION_BENCH_SCHEMA_V1: &str = "camelid.agent-orchestration-bench/v1";

/// Every agent-family schema this verifier understands.
pub const RECOGNIZED_SCHEMAS: [&str; 3] = [
    SYSCAP_RECEIPT_SCHEMA_V1,
    ORCHESTRATION_RECEIPT_SCHEMA_V1,
    ORCHESTRATION_BENCH_SCHEMA_V1,
];

/// True when `schema` names an agent-family receipt this module can verify.
/// The CLI uses this to route a receipt to this verifier instead of the parity
/// verifier.
pub fn is_agent_schema(schema: &str) -> bool {
    RECOGNIZED_SCHEMAS.contains(&schema)
}

/// Which agent gate produced a receipt. Each family has a distinct body shape
/// and a distinct honest-scope contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentReceiptClass {
    /// `camelid.agent-syscap-receipt/v1`.
    Syscap,
    /// `camelid.agent-orchestration-receipt/v1`.
    Orchestration,
    /// `camelid.agent-orchestration-bench/v1`.
    OrchestrationBench,
}

impl AgentReceiptClass {
    fn from_schema(schema: &str) -> Option<Self> {
        match schema {
            SYSCAP_RECEIPT_SCHEMA_V1 => Some(Self::Syscap),
            ORCHESTRATION_RECEIPT_SCHEMA_V1 => Some(Self::Orchestration),
            ORCHESTRATION_BENCH_SCHEMA_V1 => Some(Self::OrchestrationBench),
            _ => None,
        }
    }

    /// The field carrying the receipt's own top-line result: `outcome` for the
    /// eval gates (PASS/FAIL/INCONCLUSIVE), `verdict` for the wall-clock bench.
    fn result_field(self) -> &'static str {
        match self {
            Self::Syscap | Self::Orchestration => "outcome",
            Self::OrchestrationBench => "verdict",
        }
    }

    /// Verify the family's honest-scope invariants and return a human-readable
    /// summary of what was checked. The digest is already confirmed by the time
    /// this runs, so these guards additionally reject a *freshly sealed* but
    /// over-claiming receipt (one the digest alone would accept).
    fn check_honest_scope(self, obj: &Map<String, Value>) -> Result<String, AgentVerifyError> {
        match self {
            Self::Syscap => {
                require_false(obj, "promotes_capability")?;
                Ok("promotes_capability=false".to_string())
            }
            Self::Orchestration => {
                require_false(obj, "promotes_capability")?;
                require_false(obj, "claims_speedup")?;
                Ok("promotes_capability=false, claims_speedup=false".to_string())
            }
            Self::OrchestrationBench => {
                // The bench measures wall-clock and never carries
                // `promotes_capability`. Its honest-scope field is
                // `speedup_claimed_for`: a string naming the one measured
                // workload a speedup may be attributed to (or "none").
                let claimed = obj
                    .get("speedup_claimed_for")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentVerifyError {
                        phase: "honest-scope",
                        reason: "bench receipt is missing the string `speedup_claimed_for` \
                                 scope field"
                            .to_string(),
                    })?;
                Ok(format!("speedup_claimed_for={claimed:?}"))
            }
        }
    }
}

/// A boolean scope field that every honest receipt of the family pins to
/// `false` (e.g. `promotes_capability`, `claims_speedup`).
fn require_false(obj: &Map<String, Value>, field: &str) -> Result<(), AgentVerifyError> {
    match obj.get(field).and_then(Value::as_bool) {
        Some(false) => Ok(()),
        Some(true) => Err(AgentVerifyError {
            phase: "honest-scope",
            reason: format!(
                "receipt sets {field}=true; an agent gate promotes no capability and this \
                 field must be false"
            ),
        }),
        None => Err(AgentVerifyError {
            phase: "honest-scope",
            reason: format!("receipt is missing the boolean `{field}` scope field"),
        }),
    }
}

/// The outcome of verifying an agent-family receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentVerifyOutcome {
    /// Schema recognized, digest intact, honest-scope invariants hold.
    Verified,
    /// A verification step failed (unrecognized schema, tampered/malformed body,
    /// or an over-claiming scope field).
    NotVerified,
}

impl AgentVerifyOutcome {
    /// Process exit code: 0 verified, 1 not verified. (No divergence/lossy axes
    /// apply to an agent receipt, so the code set is deliberately just 0/1.)
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Verified => 0,
            Self::NotVerified => 1,
        }
    }
}

/// The verification step that failed, and why. `phase` is a stable short label
/// echoed in the final verdict line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentVerifyError {
    pub phase: &'static str,
    pub reason: String,
}

/// What a verified receipt attests, for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentReceiptSummary {
    pub schema: String,
    pub class: AgentReceiptClass,
    pub receipt_id: String,
    /// The receipt's own top-line result (`outcome` or `verdict`), verbatim.
    pub result: String,
    /// Human-readable summary of the honest-scope fields that were checked.
    pub scope_note: String,
    /// `os/arch` or `os/arch (hostname)` from the receipt's `host` block.
    pub host: String,
}

/// Pure verification: no I/O, no printing. Returns what the receipt attests, or
/// the first failing step. Exhaustively unit-tested.
pub fn check(value: &Value) -> Result<AgentReceiptSummary, AgentVerifyError> {
    let obj = value.as_object().ok_or_else(|| AgentVerifyError {
        phase: "parse",
        reason: "receipt is not a JSON object".to_string(),
    })?;

    // Schema — must be one this verifier understands.
    let schema = obj
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentVerifyError {
            phase: "schema",
            reason: "receipt has no string `schema` field".to_string(),
        })?;
    let class = AgentReceiptClass::from_schema(schema).ok_or_else(|| AgentVerifyError {
        phase: "schema",
        reason: format!(
            "unrecognized schema {schema:?}; this verifier understands {RECOGNIZED_SCHEMAS:?}"
        ),
    })?;

    // Receipt id — must be a 64-char lowercase-hex SHA-256.
    let receipt_id = obj
        .get("receipt_id")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentVerifyError {
            phase: "self-digest",
            reason: "receipt has no string `receipt_id` field".to_string(),
        })?;
    if !is_sha256_hex(receipt_id) {
        return Err(AgentVerifyError {
            phase: "self-digest",
            reason: format!(
                "receipt_id {receipt_id:?} is not a 64-character lowercase-hex SHA-256"
            ),
        });
    }

    // Strict self-digest over the exact document body.
    let computed = receipt_id_over(value);
    if computed != receipt_id {
        return Err(AgentVerifyError {
            phase: "self-digest",
            reason: format!(
                "receipt_id does not match the canonical body (stored {receipt_id}, computed \
                 {computed}); the receipt is tampered or malformed"
            ),
        });
    }

    // Honest-scope — reject a well-formed but over-claiming receipt.
    let scope_note = class.check_honest_scope(obj)?;

    Ok(AgentReceiptSummary {
        schema: schema.to_string(),
        class,
        receipt_id: receipt_id.to_string(),
        result: obj
            .get(class.result_field())
            .and_then(Value::as_str)
            .unwrap_or("(none)")
            .to_string(),
        scope_note,
        host: host_line(obj),
    })
}

/// Verify a parsed receipt, printing one PASS/FAIL/NOTE line per step and a
/// single final verdict, and return the outcome.
pub fn verify_value(value: &Value) -> AgentVerifyOutcome {
    match check(value) {
        Ok(summary) => {
            println!(
                "PASS schema: recognized agent-family receipt ({})",
                summary.schema
            );
            println!(
                "PASS self-digest: receipt_id matches the canonical body ({})",
                summary.receipt_id
            );
            println!(
                "PASS honest-scope: {} (this gate promotes no capability)",
                summary.scope_note
            );
            println!(
                "NOTE receipt: {}={}; host {}",
                summary.class.result_field(),
                summary.result,
                summary.host
            );
            println!("AGENT RECEIPT VERIFIED (tamper-evident; promotes no capability)");
            AgentVerifyOutcome::Verified
        }
        Err(err) => {
            println!("FAIL {}: {}", err.phase, err.reason);
            println!("AGENT RECEIPT NOT VERIFIED ({})", err.phase);
            AgentVerifyOutcome::NotVerified
        }
    }
}

/// Read a receipt file, parse it, and verify it. Read/parse failures are
/// reported as a `NotVerified` outcome (never a panic).
pub fn run(path: &Path) -> AgentVerifyOutcome {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            println!("FAIL parse: could not read {}: {err}", path.display());
            println!("AGENT RECEIPT NOT VERIFIED (parse)");
            return AgentVerifyOutcome::NotVerified;
        }
    };
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            println!("FAIL parse: receipt does not parse as JSON: {err}");
            println!("AGENT RECEIPT NOT VERIFIED (parse)");
            return AgentVerifyOutcome::NotVerified;
        }
    };
    verify_value(&value)
}

/// A 64-character lowercase-hex string, as produced by [`super::sha256_hex`].
fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `os/arch` or `os/arch (hostname)` from the receipt's `host` block, with `?`
/// standing in for any absent piece.
fn host_line(obj: &Map<String, Value>) -> String {
    let host = obj.get("host").and_then(Value::as_object);
    let field = |key: &str| {
        host.and_then(|h| h.get(key))
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string()
    };
    let os = field("os");
    let arch = field("arch");
    match host.and_then(|h| h.get("hostname")).and_then(Value::as_str) {
        Some(name) => format!("{os}/{arch} ({name})"),
        None => format!("{os}/{arch}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Stamp `receipt_id` over the body exactly as the emitters do, so the test
    /// fixtures are sealed by the same primitive the verifier checks.
    fn seal(mut body: Value) -> Value {
        let id = receipt_id_over(&body);
        body.as_object_mut()
            .expect("body is an object")
            .insert("receipt_id".to_string(), Value::String(id));
        body
    }

    fn syscap_body() -> Value {
        json!({
            "schema": SYSCAP_RECEIPT_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_759u64,
            "feature": "windows-system-control: run_windows_command (Exec) + inspect_system (Read)",
            "outcome": "PASS",
            "host": { "os": "windows", "arch": "x86_64", "hostname": "TEST-BOX" },
            "cases": [
                { "name": "echo", "input": "echo hi", "observed": "hi", "verdict": "PASS" }
            ],
            "note": "syscap tools verified under the gate",
            "promotes_capability": false
        })
    }

    fn orchestration_body() -> Value {
        json!({
            "schema": ORCHESTRATION_RECEIPT_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_760u64,
            "feature": "subagent-orchestration: spawn_subagent + check_subagent_status",
            "rung": 2,
            "outcome": "PASS",
            "host": { "os": "linux", "arch": "aarch64" },
            "cases": [ { "name": "spawn", "observed": "collected", "verdict": "PASS" } ],
            "note": "stub round-trip + caps/depth/reaping",
            "promotes_capability": false,
            "claims_speedup": false
        })
    }

    fn bench_body() -> Value {
        json!({
            "schema": ORCHESTRATION_BENCH_SCHEMA_V1,
            "receipt_id": "",
            "created_unix": 1_784_747_761u64,
            "rung": 4,
            "host": { "os": "windows", "arch": "x86_64", "hostname": "BENCH-BOX" },
            "workloads": [
                {
                    "name": "io_bound_sleep",
                    "subagents": 4,
                    "sequential_ms": 400,
                    "concurrent_ms": 120,
                    "speedup": 3.3333333333333335,
                    "note": "sleeps overlap"
                }
            ],
            "verdict": "io-bound concurrency helped; inference-bound not measured",
            "speedup_claimed_for": "io_bound_sleep",
            "notes": []
        })
    }

    #[test]
    fn recognizes_exactly_the_three_agent_schemas() {
        assert!(is_agent_schema(SYSCAP_RECEIPT_SCHEMA_V1));
        assert!(is_agent_schema(ORCHESTRATION_RECEIPT_SCHEMA_V1));
        assert!(is_agent_schema(ORCHESTRATION_BENCH_SCHEMA_V1));
        assert!(!is_agent_schema("camelid.parity-receipt/v1"));
        assert!(!is_agent_schema("camelid.agent-eval/v1"));
        assert!(!is_agent_schema(""));
    }

    #[test]
    fn verifies_a_sealed_syscap_receipt() {
        let receipt = seal(syscap_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::Syscap);
        assert_eq!(summary.result, "PASS");
        assert_eq!(summary.scope_note, "promotes_capability=false");
        assert_eq!(summary.host, "windows/x86_64 (TEST-BOX)");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn verifies_a_sealed_orchestration_receipt() {
        let receipt = seal(orchestration_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::Orchestration);
        assert_eq!(
            summary.scope_note,
            "promotes_capability=false, claims_speedup=false"
        );
        // No hostname in this fixture.
        assert_eq!(summary.host, "linux/aarch64");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn verifies_a_sealed_bench_receipt() {
        let receipt = seal(bench_body());
        let summary = check(&receipt).expect("verifies");
        assert_eq!(summary.class, AgentReceiptClass::OrchestrationBench);
        assert_eq!(
            summary.result,
            "io-bound concurrency helped; inference-bound not measured"
        );
        assert_eq!(summary.scope_note, "speedup_claimed_for=\"io_bound_sleep\"");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::Verified);
    }

    #[test]
    fn a_tampered_field_fails_the_self_digest() {
        let mut receipt = seal(syscap_body());
        // Flip the outcome after sealing; the stored receipt_id no longer matches.
        receipt["outcome"] = json!("FAIL");
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
        assert_eq!(verify_value(&receipt), AgentVerifyOutcome::NotVerified);
    }

    #[test]
    fn an_injected_extra_field_fails_the_self_digest() {
        // Strictness: the digest covers the whole document, so an added field that
        // no emitter sealed is caught (the struct-based check would silently drop
        // an unknown field — this Value-based check does not).
        let mut receipt = seal(syscap_body());
        receipt
            .as_object_mut()
            .unwrap()
            .insert("injected".to_string(), json!(true));
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
    }

    #[test]
    fn a_removed_field_fails_the_self_digest() {
        let mut receipt = seal(syscap_body());
        receipt.as_object_mut().unwrap().remove("note");
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "self-digest");
    }

    #[test]
    fn a_sealed_but_overclaiming_syscap_receipt_fails_honest_scope() {
        // Seal WITH the over-claim so the digest is intact; only the honest-scope
        // guard should reject it.
        let mut body = syscap_body();
        body["promotes_capability"] = json!(true);
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("promotes_capability=true"));
    }

    #[test]
    fn a_sealed_orchestration_speedup_claim_fails_honest_scope() {
        let mut body = orchestration_body();
        body["claims_speedup"] = json!(true);
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
        assert!(err.reason.contains("claims_speedup=true"));
    }

    #[test]
    fn a_missing_scope_field_fails_honest_scope() {
        let mut body = syscap_body();
        body.as_object_mut().unwrap().remove("promotes_capability");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
    }

    #[test]
    fn a_bench_missing_its_scope_field_fails_honest_scope() {
        let mut body = bench_body();
        body.as_object_mut().unwrap().remove("speedup_claimed_for");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "honest-scope");
    }

    #[test]
    fn an_unrecognized_schema_is_rejected() {
        let mut body = syscap_body();
        body["schema"] = json!("camelid.parity-receipt/v1");
        let receipt = seal(body);
        let err = check(&receipt).expect_err("must fail");
        assert_eq!(err.phase, "schema");
    }

    #[test]
    fn a_malformed_receipt_id_is_rejected_before_hashing() {
        for bad in [
            "",
            "tooshort",
            &"a".repeat(63),
            &"a".repeat(65),
            &"A".repeat(64), // uppercase is not our lowercase-hex form
            &format!("{}z", "a".repeat(63)),
        ] {
            let mut receipt = syscap_body();
            receipt["receipt_id"] = json!(bad);
            let err = check(&receipt).expect_err("must fail");
            assert_eq!(err.phase, "self-digest", "id {bad:?} should be rejected");
        }
    }

    #[test]
    fn a_non_object_is_rejected() {
        let err = check(&json!(["not", "an", "object"])).expect_err("must fail");
        assert_eq!(err.phase, "parse");
    }

    #[test]
    fn a_missing_schema_is_rejected() {
        let mut body = syscap_body();
        body.as_object_mut().unwrap().remove("schema");
        let err = check(&body).expect_err("must fail");
        assert_eq!(err.phase, "schema");
    }

    #[test]
    fn run_reads_and_verifies_a_receipt_file() {
        let dir = std::env::temp_dir().join(format!("camelid-agent-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let good = dir.join("good.json");
        std::fs::write(
            &good,
            serde_json::to_vec_pretty(&seal(syscap_body())).unwrap(),
        )
        .expect("write");
        assert_eq!(run(&good), AgentVerifyOutcome::Verified);

        let mut tampered_value = seal(syscap_body());
        tampered_value["outcome"] = json!("FAIL");
        let tampered = dir.join("tampered.json");
        std::fs::write(
            &tampered,
            serde_json::to_vec_pretty(&tampered_value).unwrap(),
        )
        .expect("write");
        assert_eq!(run(&tampered), AgentVerifyOutcome::NotVerified);

        let missing = dir.join("does-not-exist.json");
        assert_eq!(run(&missing), AgentVerifyOutcome::NotVerified);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
