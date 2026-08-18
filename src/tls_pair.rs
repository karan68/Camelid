//! A PEM certificate/key pair, and the one rule about it both front doors need.
//!
//! The engine's listener and the fabric proxy each accept `--tls-cert` and
//! `--tls-key`, and each has to answer the same question: what does half a pair
//! mean? It lives here rather than in either of them so the answer cannot
//! differ between the two, and so that neither has to reach into the other.
//!
//! Deliberately not a home for anything else. Loading and serving the pair is
//! `rustls`' job, through `axum_server`, in both places.

use std::io::{Error, ErrorKind, Result};
use std::path::PathBuf;

/// The paths a listener presents to its clients.
#[derive(Debug, Clone)]
pub(crate) struct TlsFiles {
    pub(crate) cert: PathBuf,
    pub(crate) key: PathBuf,
}

/// Resolve a certificate/key pair, refusing half a pair.
///
/// Half a pair is a mistake, not a request for cleartext: serving without the
/// certificate the operator clearly meant to use would be the one outcome
/// nobody asked for, so it fails rather than falls back.
pub(crate) fn resolve_tls(cert: Option<PathBuf>, key: Option<PathBuf>) -> Result<Option<TlsFiles>> {
    match (cert, key) {
        (Some(cert), Some(key)) => Ok(Some(TlsFiles { cert, key })),
        (None, None) => Ok(None),
        _ => Err(Error::new(
            ErrorKind::InvalidInput,
            "--tls-cert and --tls-key must be provided together",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_pair_resolves_and_no_pair_is_not_an_error() {
        assert!(resolve_tls(None, None)
            .expect("no pair is allowed")
            .is_none());
        let pair = resolve_tls(
            Some(PathBuf::from("certificate-chain")),
            Some(PathBuf::from("private-key")),
        )
        .expect("a complete pair resolves")
        .expect("a pair was given");
        assert_eq!(pair.cert, PathBuf::from("certificate-chain"));
        assert_eq!(pair.key, PathBuf::from("private-key"));
    }

    #[test]
    fn half_a_pair_names_both_flags_so_the_operator_knows_which_is_missing() {
        for half in [
            (Some(PathBuf::from("certificate-chain")), None),
            (None, Some(PathBuf::from("private-key"))),
        ] {
            let error = resolve_tls(half.0, half.1).expect_err("half a pair is refused");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert!(error.to_string().contains("--tls-cert"), "{error}");
            assert!(error.to_string().contains("--tls-key"), "{error}");
        }
    }
}
