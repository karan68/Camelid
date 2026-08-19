//! Whether the work a request asked for is still wanted.
//!
//! Placement and forwarding are synchronous socket I/O, and the proxy runs them
//! on a blocking thread. A blocking thread cannot be aborted from outside —
//! dropping its `JoinHandle` detaches it rather than stopping it — so the only
//! way to end that work early is to tell it, and have it agree to look.
//!
//! That matters more here than the shape of the code suggests. A Camelid node
//! executes at most one decode step at a time, so a request nobody is waiting
//! for does not merely waste a thread in this process: it holds the *node's*
//! generation slot until the node finishes or `--forward-timeout-s` expires,
//! and on a two-machine fabric that is half the operator's hardware, spent on
//! an answer that will be thrown away.
//!
//! # What is promised
//!
//! A cancelled request stops at the next socket-operation boundary, not
//! instantly. In practice that is within one read timeout (100 ms) while
//! waiting for a node to answer, and within one dial attempt while connecting.
//! Nothing here interrupts a syscall already in progress.
//!
//! One window is deliberately not covered: a *write* to the node uses the
//! whole forward budget as its timeout, so a peer that accepts the connection
//! and then stops reading can hold a send for far longer than either bound
//! above. It takes a pathological node to reach — request bodies are capped at
//! 16 MiB and a node reads its request promptly — and shortening it would mean
//! chunking every write to poll a flag between the pieces.
//!
//! Cancelling is not a failure of the node it was placed on, and the rest of
//! the fabric treats it accordingly: the observation the request was placed
//! from is kept, and the request is never sent to a second node, because there
//! is nobody left to send an answer to.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One request's cancellation, shared between whoever waits for the answer and
/// whoever is producing it.
///
/// Cloning shares the same signal rather than copying its state, so a caller
/// hands a clone to the work and keeps one to fire.
#[derive(Clone, Debug)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A cancellation for one request, not yet fired.
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// A cancellation nothing will ever fire.
    ///
    /// For callers with no client to lose: `fabric status|route|run` send one
    /// request per process and are ended by ending the process, and a health
    /// probe is already bounded by its own timeout. Saying so at those call
    /// sites keeps it a decision rather than an omission.
    pub fn never() -> Self {
        Self::new()
    }

    /// Stop wanting the answer. Idempotent, and safe to call from any thread.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether the answer has been given up on.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

impl Default for Cancel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_cancellation_has_not_been_fired() {
        assert!(!Cancel::new().is_cancelled());
        assert!(!Cancel::never().is_cancelled());
    }

    #[test]
    fn a_clone_carries_the_same_signal_rather_than_a_copy_of_it() {
        let held_by_the_caller = Cancel::new();
        let handed_to_the_work = held_by_the_caller.clone();

        held_by_the_caller.cancel();

        assert!(
            handed_to_the_work.is_cancelled(),
            "the work was not told, so it would run to its own deadline"
        );
    }

    #[test]
    fn firing_twice_is_the_same_as_firing_once() {
        let cancel = Cancel::new();
        cancel.cancel();
        cancel.cancel();
        assert!(cancel.is_cancelled());
    }
}
