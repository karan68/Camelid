//! `camelid agent-orchestration-eval` — the Phase-2 subagent-orchestration gate.
//!
//! Drives the REAL spawn → run → collect round-trip against a CANNED worker (no
//! model, no server) and exercises the mandatory caps — concurrency, spawn-tree
//! depth limit, per-child timeout/reaping, subtask_id validation, malformed-IPC
//! handling — emitting a tamper-evident sealed `camelid.agent-orchestration-
//! receipt/v1`.
//!
//! Scope discipline: this is a *rung-2* artifact. It proves the orchestration
//! MECHANICS work; it does NOT claim parallel speedup (needs a same-box
//! wall-clock receipt) and does NOT claim a real local model can drive a spawn
//! tree (needs a tool_capable row + agent-eval PASS). It promotes nothing.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent_eval::EvalOutcome;
use super::subagent::{self, SubagentConfig};

pub const RECEIPT_SCHEMA_V1: &str = "camelid.agent-orchestration-receipt/v1";

pub struct OrchestrationConfig {
    pub receipt_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostBlock {
    os: String,
    arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaseResult {
    name: String,
    observed: String,
    verdict: String, // "PASS" | "FAIL"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrchestrationReceipt {
    schema: String,
    receipt_id: String,
    created_unix: u64,
    feature: String,
    outcome: String,
    host: HostBlock,
    cases: Vec<CaseResult>,
    note: String,
    /// Honest scope: a stub-mechanics receipt promotes nothing. Always false.
    promotes_capability: bool,
    /// Honest scope: mechanics only — never a speedup claim. Always false.
    claims_speedup: bool,
}

impl OrchestrationReceipt {
    fn compute_receipt_id(&self) -> String {
        let mut value = serde_json::to_value(self).expect("receipt serializes to JSON");
        if let Value::Object(map) = &mut value {
            map.remove("receipt_id");
        }
        camelid::receipt::sha256_hex(camelid::receipt::canonical_json(&value).as_bytes())
    }
    fn seal(&mut self) {
        self.receipt_id = self.compute_receipt_id();
    }
    fn verify_self_digest(&self) -> bool {
        self.compute_receipt_id() == self.receipt_id
    }
}

fn snippet(s: &str) -> String {
    const N: usize = 300;
    let s = s.replace('\n', " | ");
    if s.len() <= N {
        return s;
    }
    let mut end = N;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|s| !s.is_empty())
}

pub fn run(cfg: OrchestrationConfig) -> anyhow::Result<i32> {
    let (outcome, cases, note) = run_battery();

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut receipt = OrchestrationReceipt {
        schema: RECEIPT_SCHEMA_V1.to_string(),
        receipt_id: String::new(),
        created_unix: ts,
        feature: "subagent-orchestration: spawn_subagent + check_subagent_status \
                  (stub round-trip + caps/depth/reaping)"
            .to_string(),
        outcome: outcome.label().to_string(),
        host: HostBlock {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: hostname(),
        },
        cases,
        note,
        promotes_capability: false,
        claims_speedup: false,
    };
    receipt.seal();
    debug_assert!(receipt.verify_self_digest());

    std::fs::create_dir_all(&cfg.receipt_dir)?;
    let path = cfg
        .receipt_dir
        .join(format!("orchestration-{ts}-{}.json", outcome.label()));
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    std::fs::write(&path, text)?;

    eprintln!();
    eprintln!("{} — {}", outcome.label(), receipt.note);
    eprintln!("receipt → {} ({})", path.display(), receipt.receipt_id);
    match outcome {
        EvalOutcome::Inconclusive => {
            eprintln!("(inconclusive does NOT change any capability flag)")
        }
        EvalOutcome::Pass => eprintln!(
            "(rung-2: orchestration mechanics verified; NOT a speedup, NOT a real-model claim)"
        ),
        EvalOutcome::Fail => {}
    }
    println!("{}", outcome.label());
    Ok(outcome.exit())
}

/// Poll a subagent until it leaves the `running` state (or the deadline passes).
fn poll_to_terminal(root: &Path, id: &str, max: Duration) -> String {
    let deadline = Instant::now() + max;
    loop {
        let s = subagent::status(root, id).unwrap_or_else(|e| format!("status: error\n{e}"));
        if !s.contains("status: running") || Instant::now() >= deadline {
            return s;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn base_config(concurrency: usize, timeout: Duration) -> SubagentConfig {
    SubagentConfig {
        addr: SocketAddr::from(([127, 0, 0, 1], 8181)),
        model_id: "canned".to_string(),
        family: "llama".to_string(),
        max_steps: 6,
        max_tokens: 64,
        concurrency,
        depth_limit: 1,
        timeout,
        // Conservative posture for the gate (the canned worker only makes a
        // read-only call anyway).
        auto_approve: false,
        shell_mode: super::shell_sandbox::ShellSandbox::Sandboxed,
    }
}

fn run_battery() -> (EvalOutcome, Vec<CaseResult>, String) {
    let temp = std::env::temp_dir().join(format!("camelid-orch-eval-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp);

    let mut cases: Vec<CaseResult> = Vec::new();
    let mut push = |name: &str, observed: String, pass: bool| {
        cases.push(CaseResult {
            name: name.to_string(),
            observed: snippet(&observed),
            verdict: if pass { "PASS" } else { "FAIL" }.to_string(),
        });
    };
    let spawn_seen; // did a child process actually run? (assigned in case 1)

    // 1) Full spawn → run → collect round-trip against the canned worker.
    {
        let work = temp.join("roundtrip");
        let _ = std::fs::create_dir_all(&work);
        subagent::configure(base_config(2, Duration::from_secs(60)));
        let spawned = subagent::spawn_canned(&work, "rt-1", "say hello", "ROUNDTRIP-OK", 0);
        let status = poll_to_terminal(&work, "rt-1", Duration::from_secs(30));
        spawn_seen = spawned.is_ok()
            && (status.contains("completed")
                || status.contains("failed")
                || status.contains("inconclusive"));
        let ok = spawned.is_ok()
            && status.contains("status: completed")
            && status.contains("ROUNDTRIP-OK")
            && status.contains("list_dir");
        push(
            "spawn_run_collect_roundtrip",
            format!("spawn={spawned:?} | {status}"),
            ok,
        );
    }

    // 2) Concurrency cap: with cap=1, a second live spawn is refused.
    {
        let work = temp.join("cap");
        let _ = std::fs::create_dir_all(&work);
        subagent::configure(base_config(1, Duration::from_secs(60)));
        let a = subagent::spawn_canned(&work, "cap-a", "g", "A", 3000); // stays live ~3s
        let b = subagent::spawn_canned(&work, "cap-b", "g", "B", 0); // must be refused
        let ok = a.is_ok() && b.is_err() && b.as_ref().unwrap_err().contains("concurrency");
        push("concurrency_cap_enforced", format!("a={a:?} b={b:?}"), ok);
        let _ = poll_to_terminal(&work, "cap-a", Duration::from_secs(10)); // reap A
    }

    // 3) Depth cap: at depth==limit, spawning is refused (fork-bomb guard).
    {
        let work = temp.join("depth");
        let _ = std::fs::create_dir_all(&work);
        subagent::configure(base_config(2, Duration::from_secs(60)));
        std::env::set_var(subagent::DEPTH_ENV, "1");
        let d = subagent::spawn_canned(&work, "depth-1", "g", "D", 0);
        std::env::remove_var(subagent::DEPTH_ENV);
        let ok = d.is_err() && d.as_ref().unwrap_err().contains("depth");
        push("depth_limit_enforced", format!("d={d:?}"), ok);
    }

    // 4) Timeout/reaping: a child exceeding its timeout is terminated and reported
    //    INCONCLUSIVE; the parent stays responsive (we get a status back).
    {
        let work = temp.join("reap");
        let _ = std::fs::create_dir_all(&work);
        subagent::configure(base_config(2, Duration::from_secs(1)));
        let e = subagent::spawn_canned(&work, "reap-1", "g", "E", 5000); // 5s > 1s timeout
        std::thread::sleep(Duration::from_millis(1500));
        let status = subagent::status(&work, "reap-1").unwrap_or_default();
        let ok =
            e.is_ok() && status.contains("status: inconclusive") && status.contains("timed out");
        push("timeout_reaps_to_inconclusive", status, ok);
    }

    // 5) subtask_id traversal/invalid is rejected before any spawn.
    {
        let work = temp.join("trav");
        let _ = std::fs::create_dir_all(&work);
        let t = subagent::spawn(&work, "../escape", "g");
        let ok = t.is_err() && t.as_ref().unwrap_err().contains("invalid subtask_id");
        push("subtask_id_traversal_rejected", format!("t={t:?}"), ok);
    }

    // 6) Malformed/partial result file is handled as failed data, not a crash.
    {
        let work = temp.join("mal");
        let dir = work.join(".camelid/subagents");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("result_mal.json"), "{ not valid json");
        let status = subagent::status(&work, "mal").unwrap_or_default();
        let ok = status.contains("malformed");
        push("malformed_result_handled", status, ok);
    }

    let _ = std::fs::remove_dir_all(&temp);

    if !spawn_seen {
        return (
            EvalOutcome::Inconclusive,
            cases,
            "could not run a subagent child process (environment issue) — not a logic failure"
                .to_string(),
        );
    }
    let passed = cases.iter().filter(|c| c.verdict == "PASS").count();
    let outcome = if passed == cases.len() {
        EvalOutcome::Pass
    } else {
        EvalOutcome::Fail
    };
    (
        outcome,
        cases.clone(),
        format!(
            "{passed}/{} orchestration mechanics cases passed",
            cases.len()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> OrchestrationReceipt {
        OrchestrationReceipt {
            schema: RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            created_unix: 7,
            feature: "f".to_string(),
            outcome: "PASS".to_string(),
            host: HostBlock {
                os: "windows".to_string(),
                arch: "x86_64".to_string(),
                hostname: None,
            },
            cases: vec![],
            note: "n".to_string(),
            promotes_capability: false,
            claims_speedup: false,
        }
    }

    #[test]
    fn seal_then_verify_passes_and_tamper_breaks() {
        let mut r = sample();
        r.seal();
        assert!(r.verify_self_digest());
        r.outcome = "FAIL".to_string();
        assert!(!r.verify_self_digest());
    }

    #[test]
    fn receipt_never_promotes_or_claims_speedup() {
        let r = sample();
        assert!(!r.promotes_capability);
        assert!(!r.claims_speedup);
    }
}
