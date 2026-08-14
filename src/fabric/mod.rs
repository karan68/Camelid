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
pub mod server;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

pub use forward::{
    wants_streaming, ForwardError, Forwarded, StreamOutcome, Streaming, DEFAULT_FORWARD_TIMEOUT,
};
pub use node::{
    parse_fabric, parse_node_spec, NodeReady, NodeSnapshot, NodeSpec, NodeSpecParseError,
    NodeStatus, DEFAULT_NODE_PORT,
};
pub use policy::{
    route, route_reserved, Reservations, RouteDecision, RouteError, RouteMode, RouteReason,
    RouteRequest,
};
pub use probe::{probe_fabric, probe_node, Observation, ProbeError, DEFAULT_PROBE_TIMEOUT};

/// How many nodes one request may be sent to before it fails.
///
/// Two means one retry. A request that finds its node gone is still served, and
/// a fabric where several nodes have gone still fails after a bounded number of
/// dials rather than walking every node it has. Raise it for a large fabric
/// where more of that walk is worth paying for; one turns failover off.
pub const DEFAULT_MAX_FORWARD_ATTEMPTS: usize = 2;

/// A configured set of nodes.
#[derive(Clone)]
pub struct Fabric {
    specs: Vec<NodeSpec>,
    timeout: Duration,
    bearer: Option<String>,
    /// Requests this fabric has placed and not yet finished.
    ///
    /// Shared across clones on purpose: the resident proxy hands a `Fabric` to
    /// every request, and they have to be counting into the same place or they
    /// cannot see each other.
    reserved: Arc<Mutex<Reservations>>,
    /// The most recent observation, reused while it is fresh enough.
    ///
    /// Shared across clones for the same reason as `reserved`: an observation
    /// only one clone can see would be re-taken by every other one.
    observed: Arc<Mutex<Option<Observation>>>,
    /// How stale a reused observation may be. Zero means never reuse one.
    max_observation_age: Duration,
    /// How many nodes one request may be sent to. One never fails over.
    max_forward_attempts: usize,
}

/// Hand-written so a token can never reach a log through a derived `Debug`,
/// the way `ServerPolicyOptions` redacts the key it holds.
impl std::fmt::Debug for Fabric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fabric")
            .field("specs", &self.specs)
            .field("timeout", &self.timeout)
            .field("bearer", &self.bearer.as_ref().map(|_| "[REDACTED]"))
            .field("max_observation_age", &self.max_observation_age)
            .field("max_forward_attempts", &self.max_forward_attempts)
            .finish()
    }
}

impl Fabric {
    pub fn new(specs: Vec<NodeSpec>) -> Self {
        Self {
            specs,
            timeout: DEFAULT_PROBE_TIMEOUT,
            bearer: None,
            reserved: Arc::new(Mutex::new(Reservations::none())),
            observed: Arc::new(Mutex::new(None)),
            max_observation_age: Duration::ZERO,
            max_forward_attempts: DEFAULT_MAX_FORWARD_ATTEMPTS,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Reuse an observation for up to `max_age` instead of probing again.
    ///
    /// A process that makes one request and exits wants the default of zero:
    /// it has nothing to reuse, and paying for the freshest possible view is
    /// free. A resident proxy is the opposite — without a bound it probes every
    /// node on every request. See [`Observation`].
    pub fn with_max_observation_age(mut self, max_age: Duration) -> Self {
        self.max_observation_age = max_age;
        self
    }

    /// How many nodes one request may be sent to before it fails.
    ///
    /// See [`DEFAULT_MAX_FORWARD_ATTEMPTS`]. One disables failover entirely and
    /// is the exact behaviour of a fabric that has never heard of it. Zero is
    /// meaningless — a request must be sent somewhere — and is read as one.
    pub fn with_max_forward_attempts(mut self, attempts: usize) -> Self {
        self.max_forward_attempts = attempts.max(1);
        self
    }

    /// Authenticate to every node with this bearer token.
    ///
    /// Required against a node started with an API key: `/v1/health` is exempt
    /// from the server's auth, so without a token such a node observes as ready
    /// and places fine, then answers the request itself with 401.
    pub fn with_bearer(mut self, bearer: Option<&str>) -> Self {
        self.bearer = bearer.map(str::to_string);
        self
    }

    pub fn specs(&self) -> &[NodeSpec] {
        &self.specs
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Observe every node.
    ///
    /// Probes unless a previous observation is still inside
    /// [`Fabric::with_max_observation_age`], which defaults to zero — so this
    /// probes every time until a caller asks for something else.
    pub fn observe(&self) -> Vec<NodeSnapshot> {
        self.observe_reporting_reuse().0
    }

    /// Observe, and say whether the answer came from a previous observation.
    ///
    /// Placement needs that second fact: a refusal decided from a reused
    /// observation is a statement about the fabric as it was, and some
    /// refusals are reported to clients as permanent.
    fn observe_reporting_reuse(&self) -> (Vec<NodeSnapshot>, bool) {
        if self.max_observation_age.is_zero() {
            return (self.probe(), false);
        }

        // Refreshing under the lock makes concurrent callers wait for one probe
        // rather than each starting their own. Without that, a burst arriving
        // on an expired observation would reproduce exactly the per-request
        // probing this bound exists to remove.
        let mut observed = lock(&self.observed);
        if let Some(observation) = observed.as_ref() {
            if observation.is_fresh_at(Instant::now(), self.max_observation_age) {
                return (observation.snapshots().to_vec(), true);
            }
        }
        let snapshots = self.probe();
        *observed = Some(Observation::taken_at(snapshots.clone(), Instant::now()));
        (snapshots, false)
    }

    /// Probe every node, whatever was observed before.
    fn probe(&self) -> Vec<NodeSnapshot> {
        probe_fabric(&self.specs, self.bearer.as_deref(), self.timeout)
    }

    /// Drop the current observation, so the next one is taken fresh.
    fn forget_observation(&self) {
        *lock(&self.observed) = None;
    }

    /// A node that did not answer may be gone, and an observation that has
    /// already proved wrong must not be reused for the rest of its window.
    ///
    /// This is what keeps the freshness bound honest: a node dying inside the
    /// window costs the one request that discovers it, not every request until
    /// the observation expires.
    fn forget_observation_if_node_vanished(&self, error: &ForwardError) {
        if matches!(
            error,
            ForwardError::Transport { .. } | ForwardError::Unreachable { .. }
        ) {
            self.forget_observation();
        }
    }

    /// Observe the fabric and choose a node for a request.
    ///
    /// The returned [`Placement`] counts the request against the chosen node
    /// until it is dropped, so a placement running concurrently can see it.
    pub fn place(&self, request: &RouteRequest<'_>) -> Result<Placement, RouteError> {
        self.place_excluding(request, &[])
    }

    /// Choose a node for a request, ignoring nodes already found gone.
    ///
    /// `excluded` holds labels this request has already been sent to and could
    /// not reach. They are dropped from the observation rather than from the
    /// policy, so placement stays the one pure decision it was.
    fn place_excluding(
        &self,
        request: &RouteRequest<'_>,
        excluded: &[String],
    ) -> Result<Placement, RouteError> {
        // Observing is socket I/O against every node. It stays outside the
        // reservation lock: holding that across it would serialise placement on
        // the slowest node, and would deadlock a `Placement` being dropped.
        let (snapshots, reused) = self.observe_reporting_reuse();

        match self.place_observed(snapshots, request, excluded) {
            // Refusing on a reused observation would settle from memory a
            // question the caller asked about now — and the proxy reports
            // `ModelUnavailable` as a 404 a client is told never to retry. A
            // node that has loaded a model since must not be refused that way,
            // so look again before saying no. This costs a probe only on a
            // request that was about to fail.
            //
            // Not once anything is excluded: by then the caller already holds a
            // forwarding failure, and that failure — not this refusal — is what
            // it will report, so a fresh look would buy nothing and would pay a
            // whole probe round against a node just found to be gone.
            Err(_) if reused && excluded.is_empty() => {
                self.forget_observation();
                let (fresh, _) = self.observe_reporting_reuse();
                self.place_observed(fresh, request, excluded)
            }
            settled => settled,
        }
    }

    /// Choose a node from an observation already taken, and reserve it.
    fn place_observed(
        &self,
        snapshots: Vec<NodeSnapshot>,
        request: &RouteRequest<'_>,
        excluded: &[String],
    ) -> Result<Placement, RouteError> {
        let snapshots: Vec<NodeSnapshot> = snapshots
            .into_iter()
            .filter(|snapshot| !excluded.iter().any(|label| label == snapshot.label()))
            .collect();

        // Deciding and recording are one step. Split them and two concurrent
        // placements both decide before either records, which is the pile-up
        // this reservation exists to prevent.
        let mut reserved = lock(&self.reserved);
        let decision = route_reserved(&snapshots, request, &reserved)?;
        reserved.take(&decision.label);
        drop(reserved);

        let node = snapshots
            .into_iter()
            .find(|snapshot| snapshot.label() == decision.label)
            .expect("placement returns a label it was given");
        Ok(Placement {
            decision,
            node,
            reserved: Arc::clone(&self.reserved),
        })
    }

    /// What this fabric currently believes it has outstanding, for tests and
    /// for reporting. A copy, so reading it holds the lock no longer than that.
    pub fn reserved(&self) -> Reservations {
        lock(&self.reserved).clone()
    }

    /// Observe, place, and send — the whole path a caller actually wants.
    ///
    /// Returns the placement alongside the answer so a caller can record which
    /// node served the request and whether affinity held. A node that turns out
    /// to be gone does not fail the request while another can serve it; see
    /// [`DEFAULT_MAX_FORWARD_ATTEMPTS`].
    pub fn dispatch(
        &self,
        path: &str,
        body: &Value,
        request: &RouteRequest<'_>,
        forward_timeout: Duration,
    ) -> Result<Dispatched, DispatchError> {
        // Refuse an unsupported request before spending any probes on it.
        forward::reject_streaming(body)?;

        let sent = self.send_until_a_node_takes_it(request, |spec| {
            forward::forward(spec, path, body, self.bearer.as_deref(), forward_timeout)
        })?;
        Ok(Dispatched {
            decision: sent.placement.decision.clone(),
            answer: sent.value,
            attempts: sent.attempts,
        })
    }

    /// Observe, place, and start a streaming request.
    ///
    /// The same placement as [`Fabric::dispatch`] — both go through
    /// [`Fabric::place`] — but the answer is read as it arrives instead of all
    /// at once. Returns once the node's response head is in, so a caller knows
    /// the status and which node it came from before relaying a single byte.
    ///
    /// The [`Placement`] comes back rather than being dropped here because the
    /// request outlives this call: the node is busy until the last event is
    /// read, so whoever pumps the stream must hold it for that long.
    pub fn dispatch_streaming(
        &self,
        path: &str,
        body: &Value,
        request: &RouteRequest<'_>,
        head_timeout: Duration,
        idle_timeout: Duration,
    ) -> Result<DispatchedStream, DispatchError> {
        let sent = self.send_until_a_node_takes_it(request, |spec| {
            forward::forward_streaming(
                spec,
                path,
                body,
                self.bearer.as_deref(),
                head_timeout,
                idle_timeout,
            )
        })?;
        Ok(DispatchedStream {
            outcome: sent.value,
            placement: sent.placement,
            attempts: sent.attempts,
        })
    }

    /// Place the request and send it, moving on while a node turns out to be
    /// gone.
    ///
    /// `send` must not have told anyone anything on the strength of an attempt
    /// it reports as failed: this calls it again against another node, which is
    /// only safe while nothing has been said and — per
    /// [`ForwardError::node_never_received_it`] — the failed node cannot have
    /// started the work.
    fn send_until_a_node_takes_it<T>(
        &self,
        request: &RouteRequest<'_>,
        mut send: impl FnMut(&NodeSpec) -> Result<T, ForwardError>,
    ) -> Result<Sent<T>, DispatchError> {
        let mut gone: Vec<String> = Vec::new();
        let mut first_failure: Option<ForwardError> = None;

        for attempt in 1..=self.max_forward_attempts.max(1) {
            let placement = match self.place_excluding(request, &gone) {
                Ok(placement) => placement,
                Err(refusal) => return Err(self.ran_out_of_nodes(first_failure, refusal)),
            };

            match send(&placement.node.spec) {
                Ok(value) => {
                    // A node this request could not reach was named by the
                    // current observation, so that observation must not outlive
                    // the request even though the request itself succeeded.
                    if !gone.is_empty() {
                        self.forget_observation();
                    }
                    return Ok(Sent {
                        value,
                        placement,
                        attempts: attempt,
                    });
                }
                Err(error) if error.node_never_received_it() => {
                    gone.push(placement.decision.label.clone());
                    first_failure.get_or_insert(error);
                }
                // The node that took the request is the one that ended it, so
                // that is what gets reported. An earlier node found gone was
                // survived — naming it would send an operator to the node the
                // fabric successfully routed around, and say nothing about the
                // one actually failing requests.
                Err(error) => {
                    if gone.is_empty() {
                        self.forget_observation_if_node_vanished(&error);
                    } else {
                        // An observation that named a node now gone must not
                        // outlive this request, however the request ended.
                        self.forget_observation();
                    }
                    return Err(DispatchError::Forward(error));
                }
            }
        }

        // The budget ran out with nodes possibly still untried. Say so with the
        // failure that started it, not with a count.
        let exhausted = first_failure.expect("the budget can only run out after a failure");
        self.forget_observation();
        Err(DispatchError::Forward(exhausted))
    }

    /// Settle on the failure to report when placement runs out of nodes.
    ///
    /// Placement can only run out on a later attempt, because this request
    /// excluded the nodes it just found gone. That finding is the story worth
    /// telling; a refusal derived from an exclusion this request invented is
    /// not. The observation is dropped with it: it named a node that is gone.
    fn ran_out_of_nodes(
        &self,
        first_failure: Option<ForwardError>,
        refusal: RouteError,
    ) -> DispatchError {
        match first_failure {
            Some(error) => {
                self.forget_observation();
                DispatchError::Forward(error)
            }
            None => DispatchError::Route(refusal),
        }
    }
}

/// A request that a node took, and what it cost to get there.
struct Sent<T> {
    value: T,
    placement: Placement,
    attempts: usize,
}

/// A complete answer from the node that served the request.
#[derive(Debug)]
pub struct Dispatched {
    pub decision: RouteDecision,
    pub answer: Forwarded,
    /// Nodes this request was sent to, the one that answered included. More
    /// than one means a node was found gone and the request was placed again.
    pub attempts: usize,
}

/// A streaming answer, still arriving from the node that took the request.
#[derive(Debug)]
pub struct DispatchedStream {
    pub outcome: StreamOutcome,
    /// Holds the node reserved; keep it until the last event is read.
    pub placement: Placement,
    /// As on [`Dispatched`]. Settled before the first byte is relayed, so it
    /// can be reported in the response head.
    pub attempts: usize,
}

/// A poisoned lock is not corrupted state: some other request panicked while
/// holding it. Recover the value rather than spreading the panic.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A node chosen for one request, and the fabric's record that it is busy.
///
/// The chosen node counts this request until this value is dropped, so hold it
/// for exactly as long as the request runs. Dropping it early tells the fabric
/// the node is free when it is not, and the next placement will pile onto it.
#[must_use = "dropping a Placement releases the node it reserved"]
pub struct Placement {
    decision: RouteDecision,
    node: NodeSnapshot,
    reserved: Arc<Mutex<Reservations>>,
}

impl Placement {
    pub fn decision(&self) -> &RouteDecision {
        &self.decision
    }

    /// The chosen node as it was observed, so a caller can read what it is
    /// already serving before building a request body for it.
    pub fn node(&self) -> &NodeSnapshot {
        &self.node
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placement")
            .field("decision", &self.decision)
            .field("node", &self.node.label())
            .finish_non_exhaustive()
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
        lock(&self.reserved).release(&self.decision.label);
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

/// Every model the fabric can serve right now, deduplicated and ordered.
///
/// Only ready nodes count. A node that is unreachable, or that has no model
/// loaded, cannot serve one — listing its model would advertise something the
/// fabric would then refuse.
///
/// Kept pure so the listing is covered by a unit test rather than by starting a
/// server, the same way [`render_status`] is.
pub fn servable_models(snapshots: &[NodeSnapshot]) -> Vec<String> {
    let mut models: Vec<String> = snapshots
        .iter()
        .filter_map(|snapshot| snapshot.active_model_id().map(str::to_string))
        .collect();
    models.sort();
    models.dedup();
    models
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
    fn the_servable_list_is_the_union_of_ready_nodes() {
        let snapshots = vec![
            snapshot("b", ready_status("zeta")),
            snapshot("a", ready_status("alpha")),
            // The same model on a second node is one entry, not two.
            snapshot("c", ready_status("alpha")),
        ];
        assert_eq!(servable_models(&snapshots), vec!["alpha", "zeta"]);
    }

    #[test]
    fn a_node_that_cannot_serve_is_not_advertised() {
        let blank = NodeStatus::Ready(NodeReady {
            active_model_id: None,
            backend: "llama".to_string(),
            version: "0.5.4".to_string(),
            in_flight: 0,
            waiting: 0,
        });
        let snapshots = vec![
            snapshot("a", ready_status("alpha")),
            snapshot(
                "dead",
                NodeStatus::Unreachable {
                    reason: "connection refused".to_string(),
                },
            ),
            snapshot(
                "empty",
                NodeStatus::NotReady {
                    reason: "no model loaded".to_string(),
                },
            ),
            snapshot("blank", blank),
        ];
        assert_eq!(servable_models(&snapshots), vec!["alpha"]);
    }

    #[test]
    fn nothing_ready_lists_nothing_rather_than_failing() {
        assert!(servable_models(&[]).is_empty());
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
    fn a_configured_token_is_never_exposed_by_debug() {
        // A `Fabric` is the kind of thing that ends up in an error context or a
        // trace line, so the derived `Debug` would have been a leak.
        let authenticated = format!("{:?}", Fabric::new(Vec::new()).with_bearer(Some("s3cret")));
        assert!(!authenticated.contains("s3cret"), "{authenticated}");
        assert!(authenticated.contains("REDACTED"), "{authenticated}");

        // Having no token reads as absent rather than as a redacted one.
        let open = format!("{:?}", Fabric::new(Vec::new()));
        assert!(open.contains("bearer: None"), "{open}");
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

    #[test]
    fn a_request_is_never_placed_on_a_node_it_has_already_failed_against() {
        let fabric = Fabric::new(Vec::new());
        let snapshots = vec![
            snapshot("a", ready_status("m")),
            snapshot("b", ready_status("m")),
        ];
        let request = RouteRequest::new(RouteMode::Throughput);

        // Ties break on label, so an unconstrained placement picks `a`; the
        // exclusion is what moves it, not a difference between the nodes.
        let first = fabric
            .place_observed(snapshots.clone(), &request, &[])
            .expect("a is eligible");
        assert_eq!(first.decision().label, "a");
        drop(first);

        let second = fabric
            .place_observed(snapshots, &request, &["a".to_string()])
            .expect("b is still eligible");
        assert_eq!(second.decision().label, "b");
    }

    #[test]
    fn excluding_every_node_refuses_rather_than_placing_on_one_anyway() {
        let fabric = Fabric::new(Vec::new());
        let snapshots = vec![snapshot("a", ready_status("m"))];
        let error = fabric
            .place_observed(
                snapshots,
                &RouteRequest::new(RouteMode::Throughput),
                &["a".to_string()],
            )
            .expect_err("nothing is left to place on");
        assert_eq!(error, RouteError::NoNodesConfigured);
    }

    #[test]
    fn a_request_must_always_be_sendable_somewhere() {
        // Zero attempts would mean a request that is never sent at all, so the
        // budget floor is part of the contract rather than a caller's problem.
        let fabric = Fabric::new(Vec::new()).with_max_forward_attempts(0);
        assert_eq!(fabric.max_forward_attempts, 1);
        assert_eq!(
            Fabric::new(Vec::new()).max_forward_attempts,
            DEFAULT_MAX_FORWARD_ATTEMPTS
        );
    }
}
