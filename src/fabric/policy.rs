//! Routing policy: which node receives a whole request.
//!
//! Every function here is pure. The fabric's correctness lives in this file, so
//! it is deliberately separated from the socket work in [`super::probe`] and can
//! be tested exhaustively against hand-built snapshots with no network, no
//! server and no model.
//!
//! Two properties are load-bearing and each has a test that fails without it:
//!
//! * **Selection is deterministic.** Ties break on label, so an operator's dry
//!   run predicts what the fabric will actually do.
//! * **Affinity degrades, never fails.** A warm prefix is an optimisation; when
//!   the node holding it dies the request still gets served, and the decision
//!   records that affinity was lost rather than silently pretending it held.

use super::node::{NodeSnapshot, NodeStatus};

/// How to choose among eligible nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    /// Spread independent requests over the least-loaded nodes.
    Throughput,
    /// Prefer the node that last served this session so its prompt prefix and KV
    /// cache stay warm, falling back to [`RouteMode::Throughput`] when it cannot.
    Affinity,
}

/// What the caller is asking the fabric to place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteRequest<'a> {
    pub mode: RouteMode,
    /// Required model id. `None` means any ready node will do.
    pub model: Option<&'a str>,
    /// Label of the node that served this session previously, if any.
    pub sticky: Option<&'a str>,
}

impl<'a> RouteRequest<'a> {
    pub fn new(mode: RouteMode) -> Self {
        Self {
            mode,
            model: None,
            sticky: None,
        }
    }

    pub fn with_model(mut self, model: Option<&'a str>) -> Self {
        self.model = model;
        self
    }

    pub fn with_sticky(mut self, sticky: Option<&'a str>) -> Self {
        self.sticky = sticky;
        self
    }
}

/// How the winning node was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteReason {
    /// The session's previous node is still eligible.
    Affinity,
    /// Exactly one node was eligible.
    OnlyCandidate,
    /// Fewest queued jobs among the eligible nodes.
    LeastLoaded,
}

impl RouteReason {
    /// The name clients see, in `fabric … --json` and in the proxy's
    /// `x-camelid-fabric-reason` header.
    ///
    /// Pinned here, and asserted in a test, because these strings used to come
    /// from `#[derive(Debug)]` — which made renaming a variant a silent change
    /// to two public surfaces.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Affinity => "Affinity",
            Self::OnlyCandidate => "OnlyCandidate",
            Self::LeastLoaded => "LeastLoaded",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub label: String,
    pub reason: RouteReason,
    /// Previous node's label when affinity was requested but could not be
    /// honoured. Reported so a caller can tell a warm hit from a cold re-prefill.
    pub affinity_lost: Option<String>,
}

/// Why no node could take the request. Every variant carries enough detail to
/// act on without re-probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    NoNodesConfigured,
    AllNodesUnavailable {
        unreachable: usize,
        not_ready: usize,
    },
    ModelUnavailable {
        model: String,
        /// What the ready nodes are actually serving.
        serving: Vec<String>,
    },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNodesConfigured => write!(f, "no nodes configured"),
            Self::AllNodesUnavailable {
                unreachable,
                not_ready,
            } => write!(
                f,
                "no node can serve: {unreachable} unreachable, {not_ready} reachable but not ready"
            ),
            Self::ModelUnavailable { model, serving } => {
                if serving.is_empty() {
                    write!(f, "no ready node is serving model `{model}`")
                } else {
                    write!(
                        f,
                        "no ready node is serving model `{model}`; ready nodes serve: {}",
                        serving.join(", ")
                    )
                }
            }
        }
    }
}

impl std::error::Error for RouteError {}

/// Requests this fabric has placed on each node and not yet finished.
///
/// `/v1/health` reports what a node has *accepted*. A request already on its way
/// to a node is invisible there until it arrives, so without this a burst of
/// concurrent requests all observe the same idle fabric, and the label
/// tie-break sends every one of them to the same node.
///
/// Pure and self-contained: the arithmetic is tested here, without threads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reservations(std::collections::BTreeMap<String, usize>);

impl Reservations {
    /// What a dry run holds: nothing is in flight, so nothing is reserved.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn get(&self, label: &str) -> usize {
        self.0.get(label).copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Record one more request placed on `label`.
    pub fn take(&mut self, label: &str) {
        *self.0.entry(label.to_string()).or_insert(0) += 1;
    }

    /// Record that one request on `label` has finished.
    ///
    /// A label that reaches zero is removed rather than left at zero, so the map
    /// stays a statement of what is actually outstanding.
    pub fn release(&mut self, label: &str) {
        if let Some(count) = self.0.get_mut(label) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.0.remove(label);
            }
        }
    }
}

/// One eligible node reduced to the fields selection actually uses.
struct Candidate<'a> {
    label: &'a str,
    load: usize,
}

/// What a node is carrying, as far as this fabric can tell. Pure.
///
/// A request this fabric placed is either not yet accepted by the node — and so
/// missing from `observed` — or already accepted and counted in it. The truth is
/// therefore somewhere in `[max(observed, reserved), observed + reserved]`, and
/// `max` is the tighter end.
///
/// Adding them instead would double-count for most of a request's life, and
/// would do so asymmetrically: a node busy with this fabric's work would be
/// ranked worse than a node equally busy with somebody else's.
fn load_of(observed: usize, reserved: usize) -> usize {
    observed.max(reserved)
}

/// Choose a node for one request, ignoring anything already in flight.
///
/// This is what a dry run does, and what `fabric route` reports: given only what
/// the nodes say about themselves, which one would take the request.
pub fn route(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
) -> Result<RouteDecision, RouteError> {
    route_reserved(snapshots, request, &Reservations::none())
}

/// Choose a node for one request, counting what this fabric already placed.
///
/// Still pure: `reserved` is a snapshot of the fabric's own bookkeeping, so the
/// decision remains a function of its arguments and a dry run with an empty
/// `reserved` reproduces [`route`] exactly.
pub fn route_reserved(
    snapshots: &[NodeSnapshot],
    request: &RouteRequest<'_>,
    reserved: &Reservations,
) -> Result<RouteDecision, RouteError> {
    if snapshots.is_empty() {
        return Err(RouteError::NoNodesConfigured);
    }

    let ready: Vec<&NodeSnapshot> = snapshots
        .iter()
        .filter(|snapshot| snapshot.status.is_ready())
        .collect();

    if ready.is_empty() {
        let unreachable = snapshots
            .iter()
            .filter(|s| matches!(s.status, NodeStatus::Unreachable { .. }))
            .count();
        return Err(RouteError::AllNodesUnavailable {
            unreachable,
            not_ready: snapshots.len() - unreachable,
        });
    }

    let serving_model: Vec<&NodeSnapshot> = match request.model {
        Some(model) => {
            let matched: Vec<&NodeSnapshot> = ready
                .iter()
                .copied()
                .filter(|snapshot| snapshot.active_model_id() == Some(model))
                .collect();
            if matched.is_empty() {
                let mut serving: Vec<String> = ready
                    .iter()
                    .filter_map(|s| s.active_model_id().map(str::to_string))
                    .collect();
                serving.sort();
                serving.dedup();
                return Err(RouteError::ModelUnavailable {
                    model: model.to_string(),
                    serving,
                });
            }
            matched
        }
        None => ready,
    };

    // Every ready node serving the model is eligible. The fabric cannot tell
    // whether a node is at its bound — `/v1/health` publishes load, not capacity
    // — so it ranks by load and lets a genuinely full node answer its own 503.
    let candidates: Vec<Candidate<'_>> = serving_model
        .iter()
        .filter_map(|snapshot| {
            snapshot.status.ready().map(|ready| Candidate {
                label: snapshot.label(),
                load: load_of(ready.in_flight, reserved.get(snapshot.label())),
            })
        })
        .collect();

    if candidates.is_empty() {
        return Err(RouteError::AllNodesUnavailable {
            unreachable: 0,
            not_ready: serving_model.len(),
        });
    }

    if request.mode == RouteMode::Affinity {
        if let Some(sticky) = request.sticky {
            if let Some(hit) = candidates.iter().find(|c| c.label == sticky) {
                return Ok(RouteDecision {
                    label: hit.label.to_string(),
                    reason: RouteReason::Affinity,
                    affinity_lost: None,
                });
            }
        }
    }

    // Ties break on label so a dry run predicts the real decision.
    let chosen = candidates
        .iter()
        .min_by(|a, b| a.load.cmp(&b.load).then_with(|| a.label.cmp(b.label)))
        .expect("candidates is non-empty");

    let reason = if candidates.len() == 1 {
        RouteReason::OnlyCandidate
    } else {
        RouteReason::LeastLoaded
    };

    let affinity_lost = request
        .sticky
        .filter(|sticky| request.mode == RouteMode::Affinity && *sticky != chosen.label)
        .map(str::to_string);

    Ok(RouteDecision {
        label: chosen.label.to_string(),
        reason,
        affinity_lost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::{NodeReady, NodeSpec, NodeStatus};

    fn spec(label: &str) -> NodeSpec {
        NodeSpec {
            label: label.to_string(),
            host: "127.0.0.1".to_string(),
            port: 8181,
        }
    }

    fn ready(label: &str, model: Option<&str>, in_flight: usize) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::Ready(NodeReady {
                active_model_id: model.map(str::to_string),
                backend: "llama".to_string(),
                version: "0.5.4".to_string(),
                in_flight,
                waiting: in_flight.saturating_sub(1),
            }),
            latency: None,
        }
    }

    fn unreachable(label: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::Unreachable {
                reason: "connection refused".to_string(),
            },
            latency: None,
        }
    }

    fn not_ready(label: &str) -> NodeSnapshot {
        NodeSnapshot {
            spec: spec(label),
            status: NodeStatus::NotReady {
                reason: "no model loaded".to_string(),
            },
            latency: None,
        }
    }

    #[test]
    fn an_empty_fabric_is_refused_distinctly_from_a_dead_one() {
        assert_eq!(
            route(&[], &RouteRequest::new(RouteMode::Throughput)),
            Err(RouteError::NoNodesConfigured)
        );
    }

    #[test]
    fn reserving_and_releasing_returns_to_nothing_outstanding() {
        let mut reserved = Reservations::none();
        assert_eq!(reserved.get("a"), 0);
        reserved.take("a");
        reserved.take("a");
        assert_eq!(reserved.get("a"), 2);
        reserved.release("a");
        assert_eq!(reserved.get("a"), 1);
        reserved.release("a");
        assert_eq!(reserved.get("a"), 0);
        // Back to genuinely empty, not a lingering zero.
        assert!(reserved.is_empty());
    }

    #[test]
    fn releasing_more_than_was_taken_cannot_underflow() {
        let mut reserved = Reservations::none();
        reserved.release("a");
        reserved.take("a");
        reserved.release("a");
        reserved.release("a");
        assert_eq!(reserved.get("a"), 0);
    }

    /// The defect this exists to fix: with every node idle, `in_flight` is 0
    /// everywhere, so the label tie-break sends a whole burst to one node.
    /// Reservations make the second request see the first.
    #[test]
    fn a_request_already_placed_moves_the_next_one_along() {
        let nodes = vec![
            ready("a", Some("m"), 0),
            ready("b", Some("m"), 0),
            ready("c", Some("m"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput);

        let mut reserved = Reservations::none();
        let mut chosen = Vec::new();
        for _ in 0..6 {
            let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
            reserved.take(&decision.label);
            chosen.push(decision.label);
        }
        // Without reservations every one of these is "a".
        assert_eq!(chosen, vec!["a", "b", "c", "a", "b", "c"]);
    }

    /// A node's own report and this fabric's bookkeeping describe overlapping
    /// work, not separate work. Summing them would rank a node carrying this
    /// fabric's request below an equally busy node carrying someone else's.
    #[test]
    fn a_nodes_own_load_and_our_reservation_are_not_added_together() {
        // `a` is busy with the one request we placed; `b` is equally busy with
        // traffic from elsewhere. They are carrying the same amount, so the tie
        // must break on label.
        let nodes = vec![ready("a", Some("m"), 1), ready("b", Some("m"), 1)];
        let mut reserved = Reservations::none();
        reserved.take("a");

        let request = RouteRequest::new(RouteMode::Throughput);
        let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
        assert_eq!(
            decision.label, "a",
            "adding load to a reservation would have made `a` look twice as busy"
        );
    }

    #[test]
    fn an_empty_reservation_set_reproduces_the_dry_run_exactly() {
        let nodes = vec![ready("a", Some("m"), 3), ready("b", Some("m"), 1)];
        let request = RouteRequest::new(RouteMode::Throughput);
        assert_eq!(
            route_reserved(&nodes, &request, &Reservations::none()),
            route(&nodes, &request)
        );
    }

    /// A reservation ranks a node, it never disqualifies one: a single reserved
    /// node is still the only candidate rather than becoming no candidate.
    #[test]
    fn reservations_never_make_a_fabric_unroutable() {
        let nodes = vec![ready("only", Some("m"), 0)];
        let mut reserved = Reservations::none();
        for _ in 0..50 {
            reserved.take("only");
        }
        let decision = route_reserved(&nodes, &RouteRequest::new(RouteMode::Throughput), &reserved)
            .expect("a reserved node is still eligible");
        assert_eq!(decision.label, "only");
        assert_eq!(decision.reason, RouteReason::OnlyCandidate);
    }

    /// Affinity is a deliberate request for one node; a reservation on it is
    /// not a reason to send the session somewhere cold.
    #[test]
    fn affinity_still_wins_over_a_reservation() {
        let nodes = vec![ready("warm", Some("m"), 0), ready("cold", Some("m"), 0)];
        let mut reserved = Reservations::none();
        reserved.take("warm");
        reserved.take("warm");

        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route_reserved(&nodes, &request, &reserved).expect("routes");
        assert_eq!(decision.label, "warm");
        assert_eq!(decision.reason, RouteReason::Affinity);
    }

    /// These strings reach clients over HTTP and in `--json` output, so they
    /// are part of the contract and a variant rename must not move them.
    #[test]
    fn the_reason_names_clients_see_are_pinned() {
        assert_eq!(RouteReason::Affinity.as_str(), "Affinity");
        assert_eq!(RouteReason::OnlyCandidate.as_str(), "OnlyCandidate");
        assert_eq!(RouteReason::LeastLoaded.as_str(), "LeastLoaded");
    }

    #[test]
    fn a_dead_fabric_reports_how_it_died() {
        let nodes = vec![unreachable("a"), not_ready("b"), unreachable("c")];
        assert_eq!(
            route(&nodes, &RouteRequest::new(RouteMode::Throughput)),
            Err(RouteError::AllNodesUnavailable {
                unreachable: 2,
                not_ready: 1,
            })
        );
    }

    #[test]
    fn an_unreachable_node_is_never_selected() {
        let nodes = vec![unreachable("a"), ready("b", None, 0)];
        let decision =
            route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("b can serve");
        assert_eq!(decision.label, "b");
        assert_eq!(decision.reason, RouteReason::OnlyCandidate);
    }

    #[test]
    fn a_wholly_idle_fabric_routes() {
        // Regression: `engine_queue_depth` is a load gauge, not a capacity bound.
        // Reading it as a bound made two healthy idle nodes both look "full" and
        // every request was refused. Verified against two live nodes.
        let nodes = vec![ready("windows", None, 0), ready("mac", None, 0)];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput))
            .expect("an idle fabric must be routable");
        assert_eq!(decision.label, "mac", "ties break on label");
    }

    #[test]
    fn the_least_loaded_eligible_node_wins() {
        let nodes = vec![
            ready("a", None, 3),
            ready("b", None, 1),
            ready("c", None, 2),
        ];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("routes");
        assert_eq!(decision.label, "b");
        assert_eq!(decision.reason, RouteReason::LeastLoaded);
    }

    #[test]
    fn ties_break_on_label_so_a_dry_run_predicts_the_real_decision() {
        let forward = vec![ready("zulu", None, 1), ready("alpha", None, 1)];
        let reversed = vec![ready("alpha", None, 1), ready("zulu", None, 1)];
        let request = RouteRequest::new(RouteMode::Throughput);
        assert_eq!(route(&forward, &request).expect("routes").label, "alpha");
        assert_eq!(route(&reversed, &request).expect("routes").label, "alpha");
    }

    #[test]
    fn a_model_request_only_matches_a_node_actually_serving_it() {
        let nodes = vec![
            ready("a", Some("llama-3b"), 0),
            ready("b", Some("qwen-4b"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("qwen-4b"));
        assert_eq!(route(&nodes, &request).expect("routes").label, "b");
    }

    #[test]
    fn a_missing_model_names_what_the_fabric_does_serve() {
        let nodes = vec![
            ready("a", Some("llama-3b"), 0),
            ready("b", Some("qwen-4b"), 0),
        ];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("gemma-27b"));
        assert_eq!(
            route(&nodes, &request),
            Err(RouteError::ModelUnavailable {
                model: "gemma-27b".to_string(),
                serving: vec!["llama-3b".to_string(), "qwen-4b".to_string()],
            })
        );
    }

    #[test]
    fn a_busy_node_still_receives_work_when_it_is_the_only_one() {
        // The fabric cannot know a node's bound, so it must not invent one and
        // refuse. A genuinely full node answers its own 503.
        let nodes = vec![ready("a", None, 99)];
        let decision = route(&nodes, &RouteRequest::new(RouteMode::Throughput)).expect("routes");
        assert_eq!(decision.label, "a");
    }

    #[test]
    fn affinity_keeps_a_session_on_its_warm_node_even_when_busier() {
        let nodes = vec![ready("warm", None, 3), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "warm");
        assert_eq!(decision.reason, RouteReason::Affinity);
        assert_eq!(decision.affinity_lost, None);
    }

    #[test]
    fn affinity_degrades_instead_of_failing_when_the_warm_node_dies() {
        let nodes = vec![unreachable("warm"), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Affinity).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("falls back rather than failing");
        assert_eq!(decision.label, "idle");
        assert_eq!(decision.affinity_lost.as_deref(), Some("warm"));
    }

    #[test]
    fn affinity_degrades_when_the_warm_node_no_longer_holds_the_model() {
        let nodes = vec![
            ready("warm", Some("llama-3b"), 0),
            ready("other", Some("qwen-4b"), 5),
        ];
        let request = RouteRequest::new(RouteMode::Affinity)
            .with_sticky(Some("warm"))
            .with_model(Some("qwen-4b"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "other");
        assert_eq!(decision.affinity_lost.as_deref(), Some("warm"));
    }

    #[test]
    fn throughput_mode_ignores_a_sticky_hint_entirely() {
        let nodes = vec![ready("warm", None, 3), ready("idle", None, 0)];
        let request = RouteRequest::new(RouteMode::Throughput).with_sticky(Some("warm"));
        let decision = route(&nodes, &request).expect("routes");
        assert_eq!(decision.label, "idle");
        // Affinity was never requested, so nothing was lost.
        assert_eq!(decision.affinity_lost, None);
    }

    #[test]
    fn a_node_with_no_model_cannot_satisfy_a_model_request() {
        let nodes = vec![ready("a", None, 0)];
        let request = RouteRequest::new(RouteMode::Throughput).with_model(Some("llama-3b"));
        assert_eq!(
            route(&nodes, &request),
            Err(RouteError::ModelUnavailable {
                model: "llama-3b".to_string(),
                serving: Vec::new(),
            })
        );
    }
}
