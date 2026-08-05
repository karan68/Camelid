//! Probing a node's `/v1/health`.
//!
//! Everything here turns one HTTP response into a routing fact. The socket work
//! lives in [`super::http`]; [`classify`] is pure and owns the decision about
//! what "ready" means.

use std::time::{Duration, Instant};

use serde::Deserialize;

use super::http::{self, HttpError};
use super::node::{NodeReady, NodeSnapshot, NodeSpec, NodeStatus};

/// Refuse a health body larger than this. A health response is a few KiB;
/// anything at this size means we are not talking to a Camelid engine.
const MAX_HEALTH_BYTES: usize = 1024 * 1024;

/// Default probe budget. Wi-Fi RTT on this fabric was measured at 3-13 ms, so
/// two seconds is generous for a healthy node and still fails a dead one fast.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    Transport(String),
    Status(u16),
    Json(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(detail) => write!(f, "{detail}"),
            Self::Status(code) => write!(f, "health endpoint answered HTTP {code}"),
            Self::Json(detail) => write!(f, "health payload was not readable: {detail}"),
        }
    }
}

impl From<HttpError> for ProbeError {
    fn from(error: HttpError) -> Self {
        Self::Transport(error.to_string())
    }
}

/// The subset of `/v1/health` the fabric routes on.
///
/// Every field but `ok` defaults, so a node running a newer or older engine that
/// renames an unrelated field still probes successfully instead of dropping out
/// of the fabric.
#[derive(Debug, Clone, Deserialize)]
struct HealthPayload {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    generation_ready: bool,
    #[serde(default)]
    active_model_id: Option<String>,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    engine_queued_tasks: usize,
    #[serde(default)]
    engine_queue_depth: usize,
}

/// Turn a successful health payload into a routing status. Pure.
fn classify(payload: &HealthPayload) -> NodeStatus {
    if !payload.ok {
        return NodeStatus::NotReady {
            reason: "engine reported not ok".to_string(),
        };
    }
    if !payload.generation_ready {
        let reason = match &payload.active_model_id {
            Some(model) => format!("model `{model}` loaded but not ready to generate"),
            None => "no model loaded".to_string(),
        };
        return NodeStatus::NotReady { reason };
    }
    NodeStatus::Ready(NodeReady {
        active_model_id: payload.active_model_id.clone(),
        backend: payload.backend.clone(),
        version: payload.version.clone(),
        // `engine_queue_depth` is a gauge of jobs in flight, not a bound; see
        // the note on `NodeReady`.
        in_flight: payload.engine_queue_depth,
        waiting: payload.engine_queued_tasks,
    })
}

fn read_health(spec: &NodeSpec, timeout: Duration) -> Result<HealthPayload, ProbeError> {
    let response = http::request(
        &spec.host,
        spec.port,
        "GET",
        "/v1/health",
        None,
        timeout,
        MAX_HEALTH_BYTES,
    )?;
    if response.status != 200 {
        return Err(ProbeError::Status(response.status));
    }
    serde_json::from_slice::<HealthPayload>(&response.body)
        .map_err(|error| ProbeError::Json(error.to_string()))
}

/// Probe one node. Never fails: an unreachable node is a routing fact, not an
/// error the caller has to handle separately.
pub fn probe_node(spec: &NodeSpec, timeout: Duration) -> NodeSnapshot {
    let started = Instant::now();
    match read_health(spec, timeout) {
        Ok(payload) => NodeSnapshot {
            spec: spec.clone(),
            status: classify(&payload),
            latency: Some(started.elapsed()),
        },
        Err(error) => NodeSnapshot {
            spec: spec.clone(),
            status: NodeStatus::Unreachable {
                reason: error.to_string(),
            },
            latency: None,
        },
    }
}

/// Probe every node concurrently, one thread each.
///
/// Sequential probing would make a fabric's status cost the sum of its timeouts,
/// so a single dead node would stall the view of every live one.
pub fn probe_fabric(specs: &[NodeSpec], timeout: Duration) -> Vec<NodeSnapshot> {
    if specs.is_empty() {
        return Vec::new();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .iter()
            .map(|spec| scope.spawn(move || probe_node(spec, timeout)))
            .collect();
        handles
            .into_iter()
            .zip(specs)
            .map(|(handle, spec)| {
                handle.join().unwrap_or_else(|_| NodeSnapshot {
                    spec: spec.clone(),
                    status: NodeStatus::Unreachable {
                        reason: "probe thread panicked".to_string(),
                    },
                    latency: None,
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(ok: bool, ready: bool, model: Option<&str>) -> HealthPayload {
        HealthPayload {
            ok,
            generation_ready: ready,
            active_model_id: model.map(str::to_string),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 1,
            engine_queue_depth: 4,
        }
    }

    #[test]
    fn a_ready_engine_becomes_a_routable_node() {
        let status = classify(&payload(true, true, Some("llama-3b")));
        let ready = status.ready().expect("ready");
        assert_eq!(ready.active_model_id.as_deref(), Some("llama-3b"));
        assert_eq!(ready.in_flight, 4);
        assert_eq!(ready.waiting, 1);
    }

    #[test]
    fn an_idle_engine_is_routable_rather_than_read_as_full() {
        // Regression: an idle node reports 0/0. Reading `engine_queue_depth` as a
        // capacity bound made a healthy idle fabric look saturated.
        let idle = HealthPayload {
            ok: true,
            generation_ready: true,
            active_model_id: Some("llama-3b".to_string()),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            engine_queued_tasks: 0,
            engine_queue_depth: 0,
        };
        let status = classify(&idle);
        assert!(status.is_ready());
        assert_eq!(status.ready().expect("ready").in_flight, 0);
    }

    #[test]
    fn a_loaded_but_unready_engine_names_the_model_it_is_warming() {
        let status = classify(&payload(true, false, Some("llama-3b")));
        match status {
            NodeStatus::NotReady { reason } => assert!(reason.contains("llama-3b"), "{reason}"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_with_no_model_is_not_ready_and_says_so() {
        match classify(&payload(true, false, None)) {
            NodeStatus::NotReady { reason } => assert_eq!(reason, "no model loaded"),
            other => panic!("expected NotReady, got {other:?}"),
        }
    }

    #[test]
    fn an_engine_reporting_not_ok_is_never_routable() {
        assert!(!classify(&payload(false, true, Some("llama-3b"))).is_ready());
    }

    #[test]
    fn a_health_payload_missing_optional_fields_still_parses() {
        // A node on a different engine version must not drop out of the fabric
        // because an unrelated field was renamed.
        let parsed: HealthPayload =
            serde_json::from_str(r#"{"ok":true,"generation_ready":true}"#).expect("parses");
        assert!(classify(&parsed).is_ready());
    }

    #[test]
    fn probing_an_empty_fabric_does_no_work() {
        assert!(probe_fabric(&[], DEFAULT_PROBE_TIMEOUT).is_empty());
    }

    #[test]
    fn an_unroutable_address_becomes_unreachable_not_an_error() {
        // Port 1 on loopback is closed; this exercises the real socket path.
        let spec = NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        };
        let snapshot = probe_node(&spec, Duration::from_millis(500));
        assert!(matches!(snapshot.status, NodeStatus::Unreachable { .. }));
        assert_eq!(snapshot.label(), "dead");
    }
}
