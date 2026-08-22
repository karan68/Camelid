//! A file the proxy re-reads while it runs, without re-reading it constantly.
//!
//! Two things the fabric loads are edited by an operator while the proxy is
//! serving — the node set and the client key set — and both want the same
//! behaviour: notice a change within a bound, do not stat the file on every
//! request, and above all **keep the last good value when a read fails**. That
//! last rule is what makes an ordinary edit safe: a file is usually replaced by
//! writing a new one and renaming it over the old, so a read landing mid-swap
//! sees a partial or missing file. Emptying the node set or refusing every
//! client on that would turn an edit into an outage.
//!
//! The cost of keeping the old value is that a change written into a broken
//! file has *not* taken effect, and nobody would know. So a failure is reported
//! as an edge — the first look that fails, and the first that succeeds again —
//! which is what lets a caller say so once rather than once per interval.
//!
//! This module owns the mechanism only. What the values mean, what is logged
//! and what a change triggers stay with the caller, because those genuinely
//! differ: a node set carries a generation that invalidates observations, and a
//! key set does not.

use std::fs;
use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

/// What the file looked like when it was last read.
///
/// Length as well as modification time, because the edits that matter here
/// usually change the length and mtime alone can be too coarse to notice a
/// rewrite within one tick. It narrows that window rather than closing it: a
/// same-length edit inside a single tick still looks unchanged. Closing it
/// would mean reading the file every interval, which is the cost this exists
/// to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    modified: Option<SystemTime>,
    len: u64,
}

impl FileStamp {
    fn of(path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        Ok(Self {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        })
    }
}

/// What a look at the file did.
///
/// Reported rather than acted on, so each caller keeps its own wording and its
/// own reaction to a change.
pub(crate) enum Change<T> {
    /// The interval had not elapsed, or the file's stamp had not moved.
    None,
    /// The file was re-read and parsed.
    Loaded {
        /// The value being replaced, so a caller can tell a real change from a
        /// file that was merely touched.
        previous: T,
        /// True when the previous look had failed, so this is the recovery.
        recovered: bool,
    },
    /// The read or parse failed and `previous` still stands.
    Failed {
        error: Error,
        /// True only on the transition into failing, so a caller can report it
        /// once instead of once per interval for as long as it lasts.
        first: bool,
    },
}

struct Watched<T> {
    value: T,
    stamp: Option<FileStamp>,
    last_checked: Option<Instant>,
    stale: bool,
}

/// A path, the value last read from it, and when it was last looked at.
pub(crate) struct WatchedFile<T> {
    path: PathBuf,
    interval: Duration,
    state: Mutex<Watched<T>>,
}

impl<T: Clone> WatchedFile<T> {
    /// Hold `value`, already loaded from `path`.
    ///
    /// The first load is the caller's, deliberately: an unusable file has to
    /// stop the proxy at startup rather than at the first request, and only the
    /// caller can say what "unusable" means for its own format.
    pub(crate) fn new(path: PathBuf, interval: Duration, value: T) -> Self {
        let stamp = FileStamp::of(&path).ok();
        Self {
            path,
            interval,
            state: Mutex::new(Watched {
                value,
                stamp,
                last_checked: Some(Instant::now()),
                stale: false,
            }),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// The last value loaded, without looking at the file again.
    pub(crate) fn cached(&self) -> T {
        lock(&self.state).value.clone()
    }

    /// The value as it stands, and what looking for a newer one did.
    ///
    /// `load` is called only when the interval has elapsed *and* the file's
    /// stamp has moved, so a busy caller pays a `stat` at most once per
    /// interval and a parse only when the file was actually written.
    ///
    /// A zero interval looks every time, which is what tests use so a change
    /// does not have to be waited out.
    pub(crate) fn look(&self, load: impl FnOnce(&Path) -> Result<T>) -> (T, Change<T>) {
        let mut state = lock(&self.state);

        let due = match state.last_checked {
            Some(checked) if self.interval > Duration::ZERO => checked.elapsed() >= self.interval,
            _ => true,
        };
        if !due {
            return (state.value.clone(), Change::None);
        }
        state.last_checked = Some(Instant::now());

        let stamp = FileStamp::of(&self.path).ok();
        // A stamp that cannot be read at all is not "unchanged" — the file may
        // have been deleted, which is a failure the caller must hear about.
        if stamp.is_some() && stamp == state.stamp {
            return (state.value.clone(), Change::None);
        }

        match load(&self.path) {
            Ok(value) => {
                let previous = std::mem::replace(&mut state.value, value);
                state.stamp = stamp;
                let recovered = state.stale;
                state.stale = false;
                (
                    state.value.clone(),
                    Change::Loaded {
                        previous,
                        recovered,
                    },
                )
            }
            Err(error) => {
                let first = !state.stale;
                state.stale = true;
                // The stamp is deliberately not updated: a file that is broken
                // now and repaired to the same length within one tick must
                // still be re-read, because the value in hand came from neither.
                (state.value.clone(), Change::Failed { error, first })
            }
        }
    }
}

/// A poisoned lock is not corrupted state: some other caller panicked while
/// holding it. Recover the value rather than spreading the panic — a proxy that
/// stopped answering every later request because one panicked is a far worse
/// failure than the one that started it.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    fn load_text(path: &Path) -> Result<String> {
        let text = fs::read_to_string(path)?;
        if text.contains("bad") {
            return Err(Error::new(ErrorKind::InvalidData, "refused"));
        }
        Ok(text)
    }

    /// Every fixture below differs in LENGTH from the one it replaces.
    ///
    /// `FileStamp` is length plus mtime, so a same-length rewrite inside one
    /// mtime tick is invisible by design. A test that changed only the content
    /// would pass or fail on how fast the machine happened to be, which is a
    /// property of the clock rather than of this module.
    const GOOD: &str = "one";
    const CHANGED: &str = "two, and a different length";
    const BROKEN: &str = "bad, and a different length";
    const REPAIRED: &str = "repaired, and a different length again";

    fn write(dir: &tempfile::TempDir, body: &str) -> PathBuf {
        let path = dir.path().join("watched");
        fs::write(&path, body).expect("write");
        path
    }

    /// Every look re-reads at a zero interval, which the tests rely on.
    fn watching(dir: &tempfile::TempDir, body: &str) -> WatchedFile<String> {
        let path = write(dir, body);
        let value = load_text(&path).expect("first load");
        WatchedFile::new(path, Duration::ZERO, value)
    }

    #[test]
    fn an_unchanged_file_is_not_parsed_again() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);

        let (value, change) = watched.look(|_| panic!("must not re-read an unchanged file"));

        assert_eq!(value, GOOD);
        assert!(matches!(change, Change::None));
    }

    #[test]
    fn a_rewritten_file_is_loaded_and_hands_back_what_it_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), CHANGED).expect("rewrite");

        let (value, change) = watched.look(load_text);

        assert_eq!(value, CHANGED);
        match change {
            Change::Loaded {
                previous,
                recovered,
            } => {
                assert_eq!(
                    previous, GOOD,
                    "the caller needs this to spot a real change"
                );
                assert!(!recovered, "nothing had failed, so nothing recovered");
            }
            _ => panic!("expected a load"),
        }
    }

    #[test]
    fn a_failed_read_keeps_the_last_good_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), BROKEN).expect("rewrite");

        let (value, change) = watched.look(load_text);

        assert_eq!(value, GOOD, "an edit must not become an outage");
        assert!(matches!(change, Change::Failed { first: true, .. }));
    }

    #[test]
    fn a_deleted_file_fails_rather_than_reading_as_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::remove_file(watched.path()).expect("remove");

        let (value, change) = watched.look(load_text);

        assert_eq!(value, GOOD);
        assert!(
            matches!(change, Change::Failed { first: true, .. }),
            "a missing file has no stamp, and must not be mistaken for an unchanged one"
        );
    }

    #[test]
    fn a_failure_is_reported_once_however_long_it_lasts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), BROKEN).expect("rewrite");

        let (_, first) = watched.look(load_text);
        assert!(matches!(first, Change::Failed { first: true, .. }));

        for _ in 0..3 {
            let (_, again) = watched.look(load_text);
            assert!(
                matches!(again, Change::Failed { first: false, .. }),
                "a file left broken would otherwise emit a line per interval forever"
            );
        }
    }

    #[test]
    fn a_repair_is_reported_as_a_recovery() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), BROKEN).expect("break it");
        watched.look(load_text);

        fs::write(watched.path(), REPAIRED).expect("repair");
        let (value, change) = watched.look(load_text);

        assert_eq!(value, REPAIRED);
        match change {
            Change::Loaded { recovered, .. } => assert!(recovered, "the operator is owed the news"),
            _ => panic!("expected a load"),
        }
    }

    /// The stamp must not advance while the file is unreadable.
    ///
    /// Tested by counting loads rather than by rewriting the file to the same
    /// length, which would only ever be a test of the clock: if a failure had
    /// advanced the stamp, the second look would find it unchanged and never
    /// call the loader at all.
    #[test]
    fn a_failed_read_does_not_advance_the_stamp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), BROKEN).expect("break it");

        let loads = std::cell::Cell::new(0_u32);
        let counting = |path: &Path| {
            loads.set(loads.get() + 1);
            load_text(path)
        };

        watched.look(counting);
        watched.look(counting);

        assert_eq!(
            loads.get(),
            2,
            "the broken file was accepted as the new stamp, so a repair of that \
             same length would never be read again"
        );
    }

    #[test]
    fn an_interval_that_has_not_elapsed_is_not_looked_at() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, GOOD);
        let watched = WatchedFile::new(path, Duration::from_secs(60), GOOD.to_string());
        fs::write(watched.path(), CHANGED).expect("rewrite");

        let (value, change) = watched.look(|_| panic!("must not look before the interval"));

        assert_eq!(value, GOOD);
        assert!(matches!(change, Change::None));
    }

    #[test]
    fn a_poisoned_lock_is_recovered_rather_than_spread() {
        let mutex = Mutex::new(1_u32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock(&mutex);
            panic!("some other caller failed");
        }));

        assert!(mutex.is_poisoned(), "the panic really did poison it");
        assert_eq!(*lock(&mutex), 1, "later callers must still be served");
    }

    /// The regression this module exists for.
    ///
    /// The state lock is held across `load`, so a panic in there poisons it.
    /// One of the two watchers this replaced took that lock with `.expect(...)`,
    /// which turned a single failed parse into a proxy that refused *every*
    /// later request — while the other recovered. Sharing one helper is what
    /// makes them agree.
    #[test]
    fn a_panic_while_loading_does_not_wedge_every_later_look() {
        let dir = tempfile::tempdir().expect("tempdir");
        let watched = watching(&dir, GOOD);
        fs::write(watched.path(), CHANGED).expect("rewrite");

        let blew_up = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            watched.look(|_| panic!("a loader panicked"));
        }));
        assert!(blew_up.is_err(), "the loader really did panic");

        let (value, _) = watched.look(load_text);
        assert_eq!(
            value, CHANGED,
            "one panicked load must not take every later caller with it"
        );
    }
}
