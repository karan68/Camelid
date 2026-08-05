//! A fabric of independent Camelid nodes.
//!
//! # Why this exists
//!
//! Camelid already has a distributed lane (`crate::distributed`) that shards one
//! model's layers across two machines. It is a *capacity* mechanism: it lets a
//! model run that fits on neither machine, and it is measurably slower than a
//! single node, because every generated token crosses the network twice and each
//! machine idles while the other computes.
//!
//! The fabric is the opposite arrangement. Each node owns a whole model and a
//! whole session; the fabric places *whole requests*. Nothing crosses the network
//! inside the token loop, so throughput scales with nodes instead of being taxed
//! by them. It composes with the layer-sharded lane rather than replacing it —
//! a fabric node may itself be a distributed coordinator.
//!
//! # Shape
//!
//! * [`node`] — identity and observed state.
//! * [`probe`] — turning `/v1/health` into a routing fact.
//! * [`policy`] — pure placement decisions; the correctness of the fabric.
//! * [`forward`] — sending a placed request to the node that will serve it.

pub mod forward;
pub(crate) mod http;
pub mod node;
pub mod policy;
pub mod probe;

use std::time::Duration;

use serde_json::Value;

pub use forward::{ForwardError, Forwarded, DEFAULT_FORWARD_TIMEOUT};
pub use node::{
    parse_fabric, parse_node_spec, NodeReady, NodeSnapshot, NodeSpec, NodeSpecParseError,
    NodeStatus, DEFAULT_NODE_PORT,
};
pub use policy::{route, RouteDecision, RouteError, RouteMode, RouteReason, RouteRequest};
pub use probe::{probe_fabric, probe_node, ProbeError, DEFAULT_PROBE_TIMEOUT};

/// A configured set of nodes.
#[derive(Debug, Clone)]
pub struct Fabric {
    specs: Vec<NodeSpec>,
    timeout: Duration,
}

impl Fabric {
    pub fn new(specs: Vec<NodeSpec>) -> Self {
        Self {
            specs,
            timeout: DEFAULT_PROBE_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn specs(&self) -> &[NodeSpec] {
        &self.specs
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Observe every node once.
    pub fn observe(&self) -> Vec<NodeSnapshot> {
        probe_fabric(&self.specs, self.timeout)
    }

    /// Observe every node once and choose one for a request.
    ///
    /// The chosen node's snapshot comes back with the decision so a caller can
    /// read what that node is already serving before it builds a request body.
    pub fn place(
        &self,
        request: &RouteRequest<'_>,
    ) -> Result<(RouteDecision, NodeSnapshot), RouteError> {
        let snapshots = self.observe();
        let decision = route(&snapshots, request)?;
        let chosen = snapshots
            .into_iter()
            .find(|snapshot| snapshot.label() == decision.label)
            .expect("placement returns a label it was given");
        Ok((decision, chosen))
    }

    /// Observe, place, and send — the whole path a caller actually wants.
    ///
    /// Returns the placement alongside the answer so a caller can record which
    /// node served the request and whether affinity held.
    pub fn dispatch(
        &self,
        path: &str,
        body: &Value,
        request: &RouteRequest<'_>,
        forward_timeout: Duration,
    ) -> Result<(RouteDecision, Forwarded), DispatchError> {
        // Refuse an unsupported request before spending any probes on it.
        forward::reject_streaming(body)?;

        let (decision, chosen) = self.place(request)?;
        let answer = forward::forward(&chosen.spec, path, body, forward_timeout)?;
        Ok((decision, answer))
    }
}

/// Why a dispatch did not produce an answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// No node could take the request.
    Route(RouteError),
    /// A node was chosen, but the request did not complete against it.
    Forward(ForwardError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Route(error) => write!(f, "{error}"),
            Self::Forward(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<RouteError> for DispatchError {
    fn from(error: RouteError) -> Self {
        Self::Route(error)
    }
}

impl From<ForwardError> for DispatchError {
    fn from(error: ForwardError) -> Self {
        Self::Forward(error)
    }
}

/// Counts of each observed state, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FabricSummary {
    pub ready: usize,
    pub not_ready: usize,
    pub unreachable: usize,
}

impl FabricSummary {
    pub fn of(snapshots: &[NodeSnapshot]) -> Self {
        let mut summary = Self::default();
        for snapshot in snapshots {
            match snapshot.status {
                NodeStatus::Ready(_) => summary.ready += 1,
                NodeStatus::NotReady { .. } => summary.not_ready += 1,
                NodeStatus::Unreachable { .. } => summary.unreachable += 1,
            }
        }
        summary
    }

    pub fn total(&self) -> usize {
        self.ready + self.not_ready + self.unreachable
    }
}

/// Render a fabric observation as fixed-width text.
///
/// Kept pure so the CLI's output is covered by a unit test rather than by
/// eyeballing a terminal.
pub fn render_status(snapshots: &[NodeSnapshot]) -> String {
    if snapshots.is_empty() {
        return "no nodes configured\n".to_string();
    }

    let label_width = snapshots
        .iter()
        .map(|s| s.label().len())
        .max()
        .unwrap_or(5)
        .max(5);
    let authority_width = snapshots
        .iter()
        .map(|s| s.spec.authority().len())
        .max()
        .unwrap_or(8)
        .max(8);

    let mut out = String::new();
    out.push_str(&format!(
        "{:<label_width$}  {:<authority_width$}  {:<9}  {:<24}  {:<6}  {}\n",
        "NODE",
        "ADDRESS",
        "STATE",
        "MODEL",
        "LOAD",
        "DETAIL",
        label_width = label_width,
        authority_width = authority_width,
    ));

    for snapshot in snapshots {
        let (state, model, load, detail) = match &snapshot.status {
            NodeStatus::Ready(ready) => (
                "ready",
                ready.active_model_id.as_deref().unwrap_or("-").to_string(),
                ready.in_flight.to_string(),
                match snapshot.latency {
                    Some(latency) => format!("{} · {} ms", ready.backend, latency.as_millis()),
                    None => ready.backend.clone(),
                },
            ),
            NodeStatus::NotReady { reason } => (
                "not-ready",
                "-".to_string(),
                "-".to_string(),
                reason.clone(),
            ),
            NodeStatus::Unreachable { reason } => {
                ("offline", "-".to_string(), "-".to_string(), reason.clone())
            }
        };
        out.push_str(&format!(
            "{:<label_width$}  {:<authority_width$}  {:<9}  {:<24}  {:<6}  {}\n",
            snapshot.label(),
            snapshot.spec.authority(),
            state,
            model,
            load,
            detail,
            label_width = label_width,
            authority_width = authority_width,
        ));
    }

    let summary = FabricSummary::of(snapshots);
    out.push_str(&format!(
        "\n{} node(s): {} ready, {} not ready, {} offline\n",
        summary.total(),
        summary.ready,
        summary.not_ready,
        summary.unreachable
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(label: &str, status: NodeStatus) -> NodeSnapshot {
        NodeSnapshot {
            spec: NodeSpec {
                label: label.to_string(),
                host: "127.0.0.1".to_string(),
                port: 8181,
            },
            status,
            latency: Some(Duration::from_millis(4)),
        }
    }

    fn ready_status(model: &str) -> NodeStatus {
        NodeStatus::Ready(NodeReady {
            active_model_id: Some(model.to_string()),
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            in_flight: 1,
            waiting: 0,
        })
    }

    #[test]
    fn an_empty_fabric_renders_a_sentence_not_an_empty_table() {
        assert_eq!(render_status(&[]), "no nodes configured\n");
    }

    #[test]
    fn the_summary_counts_every_state() {
        let snapshots = vec![
            snapshot("a", ready_status("llama-3b")),
            snapshot(
                "b",
                NodeStatus::NotReady {
                    reason: "no model loaded".to_string(),
                },
            ),
            snapshot(
                "c",
                NodeStatus::Unreachable {
                    reason: "cannot connect".to_string(),
                },
            ),
        ];
        let summary = FabricSummary::of(&snapshots);
        assert_eq!(summary.ready, 1);
        assert_eq!(summary.not_ready, 1);
        assert_eq!(summary.unreachable, 1);
        assert_eq!(summary.total(), 3);
    }

    #[test]
    fn the_table_reports_model_load_and_failure_detail() {
        let snapshots = vec![
            snapshot("windows", ready_status("llama-3b")),
            snapshot(
                "mac",
                NodeStatus::Unreachable {
                    reason: "cannot connect: refused".to_string(),
                },
            ),
        ];
        let rendered = render_status(&snapshots);
        assert!(rendered.contains("windows"), "{rendered}");
        assert!(rendered.contains("llama-3b"), "{rendered}");
        assert!(rendered.contains("LOAD"), "{rendered}");
        assert!(rendered.contains("cannot connect: refused"), "{rendered}");
        assert!(rendered.contains("2 node(s): 1 ready"), "{rendered}");
    }

    #[test]
    fn a_fabric_with_no_specs_observes_nothing() {
        let fabric = Fabric::new(Vec::new());
        assert!(fabric.is_empty());
        assert!(fabric.observe().is_empty());
    }

    #[test]
    fn dispatch_refuses_a_streaming_request_before_probing_anything() {
        // The node here is unreachable on purpose: if dispatch probed first, this
        // would surface as a routing failure instead of the streaming refusal.
        let fabric = Fabric::new(vec![NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        }]);
        let body = serde_json::json!({ "model": "m", "stream": true });
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &body,
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(200),
            )
            .expect_err("streaming is unsupported");
        assert!(
            matches!(error, DispatchError::Forward(ForwardError::Unsupported(_))),
            "got {error:?}"
        );
    }

    #[test]
    fn dispatch_on_an_empty_fabric_fails_at_placement_not_transport() {
        let fabric = Fabric::new(Vec::new());
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &serde_json::json!({ "model": "m" }),
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(200),
            )
            .expect_err("no nodes");
        assert_eq!(error, DispatchError::Route(RouteError::NoNodesConfigured));
    }

    #[test]
    fn dispatch_reports_a_dead_node_as_a_placement_failure() {
        // Every node being unreachable is a routing outcome, not a forward error:
        // there was never a node to send to.
        let fabric = Fabric::new(vec![NodeSpec {
            label: "dead".to_string(),
            host: "127.0.0.1".to_string(),
            port: 1,
        }])
        .with_timeout(Duration::from_millis(300));
        let error = fabric
            .dispatch(
                "/v1/chat/completions",
                &serde_json::json!({ "model": "m" }),
                &RouteRequest::new(RouteMode::Throughput),
                Duration::from_millis(300),
            )
            .expect_err("node is dead");
        assert!(
            matches!(
                error,
                DispatchError::Route(RouteError::AllNodesUnavailable { .. })
            ),
            "got {error:?}"
        );
    }
}
