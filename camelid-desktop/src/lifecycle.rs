//! Own a sidecar across asynchronous startup and native-window shutdown.
//! Dropping the owned value terminates and reaps the engine process.

use std::sync::Mutex;

pub struct OwnedSidecar<T>(Mutex<Slot<T>>);

struct Slot<T> {
    engine: Option<T>,
    exiting: bool,
}

impl<T> Default for OwnedSidecar<T> {
    fn default() -> Self {
        Self(Mutex::new(Slot {
            engine: None,
            exiting: false,
        }))
    }
}

impl<T> OwnedSidecar<T> {
    pub fn install(&self, engine: T) -> bool {
        let previous = {
            let mut slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
            if slot.exiting {
                // The incoming engine is dropped even if startup completed after exit.
                return false;
            }
            slot.engine.replace(engine)
        };
        // Process termination must not hold the state lock.
        drop(previous);
        true
    }

    pub fn stop(&self, exiting: bool) {
        let engine = {
            let mut slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
            slot.exiting |= exiting;
            slot.engine.take()
        };
        drop(engine);
    }
}

#[cfg(test)]
mod tests {
    use super::OwnedSidecar;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Sidecar(Arc<AtomicUsize>);
    impl Drop for Sidecar {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn exit_stops_the_engine_once_and_rejects_late_startup() {
        let stops = Arc::new(AtomicUsize::new(0));
        let state = OwnedSidecar::default();
        assert!(state.install(Sidecar(stops.clone())));
        state.stop(true);
        state.stop(true);
        assert_eq!(stops.load(Ordering::SeqCst), 1);
        assert!(!state.install(Sidecar(stops.clone())));
        assert_eq!(stops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn closing_during_startup_cannot_be_reversed_by_retry() {
        let stops = Arc::new(AtomicUsize::new(0));
        let state = OwnedSidecar::default();
        state.stop(true);
        state.stop(false);
        assert!(!state.install(Sidecar(stops.clone())));
        assert_eq!(stops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retry_can_start_a_replacement_before_exit() {
        let stops = Arc::new(AtomicUsize::new(0));
        let state = OwnedSidecar::default();
        assert!(state.install(Sidecar(stops.clone())));
        state.stop(false);
        assert_eq!(stops.load(Ordering::SeqCst), 1);
        assert!(state.install(Sidecar(stops.clone())));
        drop(state);
        assert_eq!(stops.load(Ordering::SeqCst), 2);
    }
}
