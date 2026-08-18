//! Which clients this proxy serves, and how one of them stops being served.
//!
//! The proxy shipped with a single key: every client that could reach it held
//! the same secret, so there was no way to tell two callers apart in a log and
//! no way to cut one off without cutting off all of them and restarting.
//!
//! A key set names its clients. The name is what the access log records — the
//! key itself never is — and removing an entry from the file revokes that
//! client without stopping the proxy or disturbing anyone else.
//!
//! # What is reused rather than rebuilt
//!
//! Nothing here compares secrets. Each client holds the engine's own
//! [`ApiAuth`], so a presented credential is parsed from the same two headers
//! and compared with the same constant-time check the engine applies to its
//! own listener, and keys are validated by the engine's rules through
//! [`crate::api::resolve_api_key`]. This module decides *who*, never *how*.

use std::collections::HashSet;
use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use axum::http::HeaderMap;
use serde::Deserialize;

use crate::api::ApiAuth;

/// Longest client name this proxy will accept.
///
/// Bounded because the name is written to every log line the client causes.
const MAX_CLIENT_NAME_BYTES: usize = 64;

/// How long a loaded key set is trusted before the file is looked at again.
///
/// Revocation is only as fast as this. A second is short enough that an
/// operator does not wait on it and long enough that a busy proxy is not
/// stat-ing a file on every request.
pub const DEFAULT_KEY_RELOAD_INTERVAL: Duration = Duration::from_secs(1);

/// What the proxy decided about a request's credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Admission {
    /// No key is configured, so the request is served and no client is named.
    Open,
    /// A configured client presented its key.
    Client(Arc<str>),
    /// A key is required and this request did not present a usable one.
    Refused,
}

/// One client, and the name its requests are recorded under.
#[derive(Clone)]
struct ClientKey {
    name: Arc<str>,
    /// The engine's single-key primitive, one per client: this is what makes
    /// the comparison here identical to the engine's own.
    auth: ApiAuth,
}

/// The on-disk shape of a key set.
#[derive(Debug, Deserialize)]
struct KeyFile {
    clients: Vec<KeyFileEntry>,
}

#[derive(Debug, Deserialize)]
struct KeyFileEntry {
    name: String,
    key: String,
}

/// What the file looked like when it was last read.
///
/// Length as well as modification time, because a revocation usually shortens
/// the file and mtime alone can be too coarse to notice a rewrite within one
/// tick. It narrows that window rather than closing it: a key swapped for one
/// of the same length inside a single tick still looks unchanged. Closing it
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

struct ReloadState {
    clients: Arc<[ClientKey]>,
    stamp: Option<FileStamp>,
    last_checked: Option<Instant>,
    /// Whether the last re-read failed, so the notice below is printed once
    /// per transition rather than once per request.
    stale: bool,
}

enum Source {
    /// A single key given on the command line or in a one-key file. It cannot
    /// change while the process runs, so there is nothing to re-read.
    Fixed(Arc<[ClientKey]>),
    /// A key set on disk, re-read when it changes.
    File {
        path: PathBuf,
        interval: Duration,
        state: Mutex<ReloadState>,
    },
}

/// The credential a client must present to this proxy.
#[derive(Clone)]
pub struct ClientAuth {
    /// `None` accepts every client, which [`super::server::bind`] only permits
    /// on a loopback listener or with the risk acknowledged.
    source: Option<Arc<Source>>,
}

impl std::fmt::Debug for ClientAuth {
    /// Never renders a key, and not the names either: this is printed by
    /// `ServeConfig`'s derive, and who may call is not something to scatter
    /// through logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientAuth")
            .field("enabled", &self.is_enabled())
            .field("reloadable", &self.is_reloadable())
            .finish()
    }
}

impl ClientAuth {
    /// Accept every client.
    pub fn none() -> Self {
        Self { source: None }
    }

    /// Require one key, read from a value or a file.
    ///
    /// Fails rather than starting unauthenticated: a proxy that silently
    /// ignored an unreadable key file would be open to whatever can reach it.
    pub fn resolve(api_key: Option<String>, api_key_file: Option<PathBuf>) -> Result<Self> {
        let Some(key) = crate::api::resolve_api_key(api_key, api_key_file)? else {
            return Ok(Self::none());
        };
        Ok(Self {
            source: Some(Arc::new(Source::Fixed(Arc::from(vec![ClientKey {
                name: Arc::from("default"),
                auth: ApiAuth::new(Some(key)),
            }])))),
        })
    }

    /// Require a key from a named set on disk, re-read as it changes.
    pub fn from_key_file(path: PathBuf) -> Result<Self> {
        Self::from_key_file_every(path, DEFAULT_KEY_RELOAD_INTERVAL)
    }

    /// [`Self::from_key_file`] with the staleness bound supplied.
    ///
    /// A zero interval re-reads on every request, which is what the tests use
    /// so a revocation does not have to be waited out.
    pub fn from_key_file_every(path: PathBuf, interval: Duration) -> Result<Self> {
        // Loaded once here so an unusable file stops the proxy at startup
        // rather than at the first request.
        let clients = load_key_file(&path)?;
        let stamp = FileStamp::of(&path).ok();
        Ok(Self {
            source: Some(Arc::new(Source::File {
                path,
                interval,
                state: Mutex::new(ReloadState {
                    clients,
                    stamp,
                    last_checked: Some(Instant::now()),
                    stale: false,
                }),
            })),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.source.is_some()
    }

    /// Whether revoking a client takes effect without a restart.
    pub fn is_reloadable(&self) -> bool {
        matches!(self.source.as_deref(), Some(Source::File { .. }))
    }

    /// How many clients are currently served. Reported at startup, never per
    /// request.
    pub fn client_count(&self) -> usize {
        match self.source.as_deref() {
            None => 0,
            Some(Source::Fixed(clients)) => clients.len(),
            Some(Source::File { state, .. }) => state.lock().expect("client keys").clients.len(),
        }
    }

    /// Decide whether `headers` may be served, and by which client's name.
    pub(crate) fn admit(&self, headers: &HeaderMap) -> Admission {
        let Some(source) = self.source.as_deref() else {
            return Admission::Open;
        };
        let clients = match source {
            Source::Fixed(clients) => Arc::clone(clients),
            Source::File {
                path,
                interval,
                state,
            } => current_clients(path, *interval, state),
        };
        match_client(&clients, headers)
    }
}

impl Default for ClientAuth {
    fn default() -> Self {
        Self::none()
    }
}

/// The accepting client, compared against every entry.
///
/// At most one entry can match, because the loader refuses a set that gives
/// two clients the same key. Stopping at the first match would therefore save
/// nothing except comparisons against a handful of remaining keys, while
/// making the time a decision takes depend on where in the operator's file a
/// client happens to sit.
fn match_client(clients: &[ClientKey], headers: &HeaderMap) -> Admission {
    let mut admitted: Option<Arc<str>> = None;
    for client in clients {
        if client.auth.accepts(headers) {
            admitted = Some(Arc::clone(&client.name));
        }
    }
    admitted.map_or(Admission::Refused, Admission::Client)
}

/// The key set as it stands, re-reading the file if it may have changed.
fn current_clients(
    path: &Path,
    interval: Duration,
    state: &Mutex<ReloadState>,
) -> Arc<[ClientKey]> {
    let mut state = state.lock().expect("client keys");

    let due = match state.last_checked {
        Some(checked) if interval > Duration::ZERO => checked.elapsed() >= interval,
        Some(_) => true,
        None => true,
    };
    if !due {
        return Arc::clone(&state.clients);
    }
    state.last_checked = Some(Instant::now());

    let stamp = FileStamp::of(path).ok();
    if stamp.is_some() && stamp == state.stamp {
        return Arc::clone(&state.clients);
    }

    match load_key_file(path) {
        Ok(clients) => {
            tracing::info!(
                clients = clients.len(),
                path = %path.display(),
                "client key set reloaded"
            );
            if state.stale {
                eprintln!(
                    "fabric: client keys reloaded from {}; now serving {}",
                    path.display(),
                    clients_phrase(clients.len())
                );
            }
            state.clients = clients;
            state.stamp = stamp;
            state.stale = false;
        }
        Err(error) => {
            // The previous set is kept on purpose. A key file is often replaced
            // by writing a new one and renaming it over the old, so a read that
            // lands mid-swap sees a partial or missing file; refusing every
            // client on that would turn an ordinary edit into an outage. The
            // cost is that a revocation written into a broken or deleted file
            // has not taken effect, so the operator has to hear about it.
            tracing::warn!(
                path = %path.display(),
                %error,
                "could not reload client keys; the previous set is still in force"
            );
            // Printed, not only traced: `RUST_LOG` is unset on a stock proxy,
            // so a revocation that has silently not happened would otherwise
            // be invisible to the operator who just wrote it. Once per
            // transition, so a file left broken does not fill the terminal.
            if !state.stale {
                eprintln!("{}", stale_key_set_notice(&error, state.clients.len()));
            }
            state.stale = true;
        }
    }
    Arc::clone(&state.clients)
}

/// What an operator is told when a re-read fails. Pure, so it is tested rather
/// than eyeballed.
///
/// It has to say the revocation did not happen, because that is the whole
/// consequence: the file on disk and the set being enforced no longer agree,
/// and the file is the one the operator is looking at.
fn stale_key_set_notice(error: &Error, clients: usize) -> String {
    format!(
        "fabric: could not reload client keys: {error}. The previous set of {} \
         is still in force, so a revocation written here has NOT taken effect.",
        clients_phrase(clients)
    )
}

fn clients_phrase(count: usize) -> String {
    if count == 1 {
        "1 client".to_string()
    } else {
        format!("{count} clients")
    }
}

/// Read and validate a key set.
fn load_key_file(path: &Path) -> Result<Arc<[ClientKey]>> {
    let text = fs::read_to_string(path).map_err(|error| {
        Error::new(
            error.kind(),
            format!("could not read client key file {}: {error}", path.display()),
        )
    })?;
    let parsed: KeyFile = serde_json::from_str(&text).map_err(|error| {
        Error::new(
            ErrorKind::InvalidData,
            format!(
                "client key file {} is not valid JSON of the form \
                 {{\"clients\":[{{\"name\":\"...\",\"key\":\"...\"}}]}}: {error}",
                path.display()
            ),
        )
    })?;

    if parsed.clients.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "client key file {} lists no clients; remove --client-keys to \
                 serve without authentication, which the bind guard will then \
                 make you acknowledge",
                path.display()
            ),
        ));
    }

    let mut names = HashSet::new();
    let mut keys = HashSet::new();
    let mut clients = Vec::with_capacity(parsed.clients.len());
    for entry in parsed.clients {
        let name = validate_client_name(&entry.name, path)?;
        // Validated by the engine's own rules, so the two front doors cannot
        // disagree about what a usable key is.
        let key = crate::api::resolve_api_key(Some(entry.key), None)?.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("client {name} in {} has no key", path.display()),
            )
        })?;

        if !names.insert(name.clone()) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "client key file {} names {name} more than once; a name has \
                     to identify one client for a log line to mean anything",
                    path.display()
                ),
            ));
        }
        if !keys.insert(key.clone()) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "client key file {} gives the same key to more than one \
                     client; revoking either would not revoke the other",
                    path.display()
                ),
            ));
        }

        clients.push(ClientKey {
            name: Arc::from(name.as_str()),
            auth: ApiAuth::new(Some(key)),
        });
    }
    Ok(Arc::from(clients))
}

/// A name that can be read back out of a log line unambiguously.
fn validate_client_name(name: &str, path: &Path) -> Result<String> {
    let trimmed = name.trim();
    let usable = !trimmed.is_empty()
        && trimmed.len() <= MAX_CLIENT_NAME_BYTES
        && trimmed.chars().all(|c| c.is_ascii_graphic());
    if !usable {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "client key file {} has a name that is empty, longer than \
                 {MAX_CLIENT_NAME_BYTES} bytes, or not printable ASCII without \
                 spaces; the name is written to every log line that client causes",
                path.display()
            ),
        ));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn bearer(key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}")).expect("header"),
        );
        headers
    }

    fn write_keys(dir: &tempfile::TempDir, body: &str) -> PathBuf {
        let path = dir.path().join("clients.json");
        fs::write(&path, body).expect("write key file");
        path
    }

    const TWO_CLIENTS: &str = r#"{"clients":[
        {"name":"laptop","key":"laptop-secret"},
        {"name":"ci","key":"ci-secret"}
    ]}"#;

    #[test]
    fn no_key_admits_everyone_and_names_nobody() {
        let auth = ClientAuth::none();
        assert!(!auth.is_enabled());
        assert_eq!(auth.admit(&HeaderMap::new()), Admission::Open);
    }

    #[test]
    fn each_client_is_admitted_under_its_own_name() {
        let dir = tempfile::tempdir().expect("temp dir");
        let auth = ClientAuth::from_key_file(write_keys(&dir, TWO_CLIENTS)).expect("load");

        assert_eq!(auth.client_count(), 2);
        assert_eq!(
            auth.admit(&bearer("laptop-secret")),
            Admission::Client(Arc::from("laptop"))
        );
        assert_eq!(
            auth.admit(&bearer("ci-secret")),
            Admission::Client(Arc::from("ci"))
        );
        assert_eq!(auth.admit(&bearer("neither")), Admission::Refused);
        assert_eq!(auth.admit(&HeaderMap::new()), Admission::Refused);
    }

    /// The point of naming clients: one stops being served, the rest do not
    /// notice, and nothing restarts.
    #[test]
    fn removing_a_client_revokes_only_that_client() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_keys(&dir, TWO_CLIENTS);
        let auth = ClientAuth::from_key_file_every(path.clone(), Duration::ZERO).expect("load");

        assert_eq!(
            auth.admit(&bearer("laptop-secret")),
            Admission::Client(Arc::from("laptop"))
        );

        fs::write(&path, r#"{"clients":[{"name":"ci","key":"ci-secret"}]}"#).expect("rewrite");

        assert_eq!(
            auth.admit(&bearer("laptop-secret")),
            Admission::Refused,
            "a removed client is still being served"
        );
        assert_eq!(
            auth.admit(&bearer("ci-secret")),
            Admission::Client(Arc::from("ci")),
            "revoking one client disturbed another"
        );
    }

    /// A file being replaced is briefly unreadable. Refusing everyone on that
    /// would turn an ordinary edit into an outage.
    #[test]
    fn a_broken_file_leaves_the_previous_set_in_force() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_keys(&dir, TWO_CLIENTS);
        let auth = ClientAuth::from_key_file_every(path.clone(), Duration::ZERO).expect("load");

        fs::write(&path, "{ not json").expect("corrupt");
        assert_eq!(
            auth.admit(&bearer("laptop-secret")),
            Admission::Client(Arc::from("laptop"))
        );

        fs::remove_file(&path).expect("remove");
        assert_eq!(
            auth.admit(&bearer("ci-secret")),
            Admission::Client(Arc::from("ci"))
        );

        // ...and a file that becomes valid again is picked up.
        fs::write(&path, r#"{"clients":[{"name":"ci","key":"ci-secret"}]}"#).expect("restore");
        assert_eq!(auth.admit(&bearer("laptop-secret")), Admission::Refused);
    }

    /// The previous set staying in force is only safe if the operator is told,
    /// and `RUST_LOG` is unset on a stock proxy, so the notice is printed. It
    /// has to say the revocation did not happen; "could not reload" alone
    /// reads like a retry that will sort itself out.
    #[test]
    fn the_notice_says_the_revocation_has_not_taken_effect() {
        let error = Error::new(ErrorKind::InvalidData, "clients.json is not valid JSON");
        let notice = stale_key_set_notice(&error, 2);
        assert!(
            notice.contains("clients.json is not valid JSON"),
            "{notice}"
        );
        assert!(notice.contains("previous set of 2 clients"), "{notice}");
        assert!(notice.contains("NOT taken effect"), "{notice}");
        assert!(
            stale_key_set_notice(&error, 1).contains("previous set of 1 client "),
            "one client is not '1 clients'"
        );
    }

    /// The set is only re-read once the bound has passed, so a busy proxy is
    /// not stat-ing the file on every request.
    #[test]
    fn a_set_inside_the_bound_is_not_re_read() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_keys(&dir, TWO_CLIENTS);
        let auth =
            ClientAuth::from_key_file_every(path.clone(), Duration::from_secs(3600)).expect("load");

        fs::write(&path, r#"{"clients":[{"name":"ci","key":"ci-secret"}]}"#).expect("rewrite");

        assert_eq!(
            auth.admit(&bearer("laptop-secret")),
            Admission::Client(Arc::from("laptop")),
            "the file was re-read before its staleness bound had passed"
        );
    }

    #[test]
    fn an_unusable_set_stops_the_proxy_rather_than_opening_it() {
        let dir = tempfile::tempdir().expect("temp dir");

        for (label, body) in [
            ("not json", "{ nope"),
            ("no clients", r#"{"clients":[]}"#),
            (
                "duplicate name",
                r#"{"clients":[{"name":"a","key":"one"},{"name":"a","key":"two"}]}"#,
            ),
            (
                "shared key",
                r#"{"clients":[{"name":"a","key":"same"},{"name":"b","key":"same"}]}"#,
            ),
            ("empty name", r#"{"clients":[{"name":"   ","key":"one"}]}"#),
            (
                "unprintable name",
                "{\"clients\":[{\"name\":\"a\\tb\",\"key\":\"one\"}]}",
            ),
            ("empty key", r#"{"clients":[{"name":"a","key":"  "}]}"#),
        ] {
            let path = write_keys(&dir, body);
            ClientAuth::from_key_file(path)
                .err()
                .unwrap_or_else(|| panic!("{label} was accepted"));
        }

        ClientAuth::from_key_file(dir.path().join("absent.json"))
            .expect_err("a missing file is not an open proxy");
    }

    /// A single `--api-key` still works, and gets a name so the log can say
    /// something other than "-".
    #[test]
    fn a_single_key_is_one_client_that_cannot_be_reloaded() {
        let auth = ClientAuth::resolve(Some("solo-secret".to_string()), None).expect("resolve");
        assert!(auth.is_enabled());
        assert!(!auth.is_reloadable());
        assert_eq!(auth.client_count(), 1);
        assert_eq!(
            auth.admit(&bearer("solo-secret")),
            Admission::Client(Arc::from("default"))
        );
        assert_eq!(auth.admit(&bearer("wrong")), Admission::Refused);
    }

    #[test]
    fn a_key_set_reports_itself_as_reloadable() {
        let dir = tempfile::tempdir().expect("temp dir");
        let auth = ClientAuth::from_key_file(write_keys(&dir, TWO_CLIENTS)).expect("load");
        assert!(auth.is_enabled());
        assert!(auth.is_reloadable());
    }

    /// Both front doors read a credential the same way, because both use the
    /// engine's [`ApiAuth`] to do it.
    #[test]
    fn either_header_the_engine_accepts_names_the_same_client() {
        let dir = tempfile::tempdir().expect("temp dir");
        let auth = ClientAuth::from_key_file(write_keys(&dir, TWO_CLIENTS)).expect("load");

        let mut api_key = HeaderMap::new();
        api_key.insert("x-api-key", HeaderValue::from_static("ci-secret"));

        assert_eq!(
            auth.admit(&api_key),
            Admission::Client(Arc::from("ci")),
            "X-API-Key must name the same client Authorization: Bearer does"
        );
        assert_eq!(auth.admit(&bearer("ci-secret")), auth.admit(&api_key));
    }

    /// A client is admitted under its own name wherever it sits in the set.
    #[test]
    fn a_clients_position_in_the_set_does_not_change_the_answer() {
        let dir = tempfile::tempdir().expect("temp dir");
        for (position, body) in [
            (
                "first",
                r#"{"clients":[{"name":"target","key":"target-secret"},
                    {"name":"b","key":"b-secret"},{"name":"c","key":"c-secret"}]}"#,
            ),
            (
                "middle",
                r#"{"clients":[{"name":"b","key":"b-secret"},
                    {"name":"target","key":"target-secret"},{"name":"c","key":"c-secret"}]}"#,
            ),
            (
                "last",
                r#"{"clients":[{"name":"b","key":"b-secret"},
                    {"name":"c","key":"c-secret"},{"name":"target","key":"target-secret"}]}"#,
            ),
        ] {
            let auth = ClientAuth::from_key_file(write_keys(&dir, body)).expect("load");
            assert_eq!(
                auth.admit(&bearer("target-secret")),
                Admission::Client(Arc::from("target")),
                "answered differently when the client was {position} in the set"
            );
            assert_eq!(
                auth.admit(&bearer("absent-secret")),
                Admission::Refused,
                "an unknown key was admitted when the target was {position}"
            );
        }
    }
}
