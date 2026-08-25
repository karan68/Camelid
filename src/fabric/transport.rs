//! Authentication and confidentiality policy for the proxy-to-node hop.
//!
//! A node already authenticates the fabric with its API bearer. This module
//! binds that application credential to an authenticated transport: either a
//! CA-pinned TLS connection, or cleartext that resolves only to loopback (the
//! supported shape for a local node or an operator-owned tunnel). Direct
//! cleartext to another machine requires an explicit acknowledgement.

use std::fmt;
use std::io::{Error, ErrorKind, Result};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::CertificateDer;

#[derive(Clone)]
pub(crate) enum NodeTransport {
    Plaintext { allow_remote: bool },
    Tls { config: Arc<ClientConfig> },
}

impl NodeTransport {
    pub(crate) fn resolve(ca_file: Option<&Path>, allow_cleartext_remote: bool) -> Result<Self> {
        match (ca_file, allow_cleartext_remote) {
            (Some(_), true) => Err(Error::new(
                ErrorKind::InvalidInput,
                "--node-tls-ca and --allow-cleartext-node-transport cannot be combined",
            )),
            (Some(path), false) => Self::from_ca_file(path),
            (None, allow_remote) => Ok(Self::Plaintext { allow_remote }),
        }
    }

    fn from_ca_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            Error::new(
                error.kind(),
                format!(
                    "could not read node TLS CA bundle `{}`: {error}",
                    path.display()
                ),
            )
        })?;
        let certificates = CertificateDer::pem_slice_iter(&bytes)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "node TLS CA bundle `{}` is not valid PEM: {error}",
                        path.display()
                    ),
                )
            })?;
        if certificates.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "node TLS CA bundle `{}` contains no certificates",
                    path.display()
                ),
            ));
        }

        let mut roots = RootCertStore::empty();
        for certificate in certificates {
            roots.add(certificate).map_err(|error| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "node TLS CA bundle `{}` contains an unusable certificate: {error}",
                        path.display()
                    ),
                )
            })?;
        }
        let config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        Ok(Self::Tls {
            config: Arc::new(config),
        })
    }

    pub(crate) fn permitted_addresses(&self, addresses: &[SocketAddr]) -> Result<Vec<SocketAddr>> {
        match self {
            Self::Tls { .. } | Self::Plaintext { allow_remote: true } => Ok(addresses.to_vec()),
            Self::Plaintext {
                allow_remote: false,
            } => {
                let loopback: Vec<_> = addresses
                    .iter()
                    .copied()
                    .filter(|address| address.ip().is_loopback())
                    .collect();
                if loopback.is_empty() {
                    return Err(Error::new(
                        ErrorKind::PermissionDenied,
                        "refusing cleartext transport to a non-loopback node; configure --node-tls-ca or acknowledge --allow-cleartext-node-transport",
                    ));
                }
                Ok(loopback)
            }
        }
    }

    pub(crate) fn tls_config(&self) -> Option<Arc<ClientConfig>> {
        match self {
            Self::Plaintext { .. } => None,
            Self::Tls { config } => Some(Arc::clone(config)),
        }
    }

    pub(crate) fn description(&self) -> &'static str {
        match self {
            Self::Tls { .. } => "server-authenticated TLS (pinned CA)",
            Self::Plaintext { allow_remote: true } => {
                "cleartext (direct remote transport explicitly acknowledged)"
            }
            Self::Plaintext {
                allow_remote: false,
            } => "cleartext restricted to loopback/tunnels",
        }
    }
}

impl Default for NodeTransport {
    fn default() -> Self {
        Self::Plaintext {
            allow_remote: false,
        }
    }
}

impl fmt::Debug for NodeTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plaintext { allow_remote } => formatter
                .debug_struct("Plaintext")
                .field("allow_remote", allow_remote)
                .finish(),
            Self::Tls { .. } => formatter.write_str("Tls { server_auth: pinned_ca }"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn guarded_cleartext_keeps_loopback_and_refuses_only_remote_answers() {
        let transport = NodeTransport::default();
        let v4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8181);
        let v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8181);
        let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10)), 8181);

        assert_eq!(
            transport
                .permitted_addresses(&[remote, v6, v4])
                .expect("loopback answers remain usable"),
            vec![v6, v4]
        );
        let error = transport
            .permitted_addresses(&[remote])
            .expect_err("remote cleartext is refused");
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert!(error.to_string().contains("--node-tls-ca"), "{error}");
    }

    #[test]
    fn cleartext_acknowledgement_allows_remote_addresses() {
        let transport = NodeTransport::resolve(None, true).expect("acknowledgement resolves");
        let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10)), 8181);
        assert_eq!(
            transport
                .permitted_addresses(&[remote])
                .expect("remote address is acknowledged"),
            vec![remote]
        );
        assert!(transport.description().contains("acknowledged"));
    }

    #[test]
    fn tls_and_a_cleartext_acknowledgement_are_not_two_simultaneous_modes() {
        let error = NodeTransport::resolve(Some(Path::new("node-ca")), true)
            .expect_err("contradictory transport flags are refused");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--node-tls-ca"), "{error}");
        assert!(
            error
                .to_string()
                .contains("--allow-cleartext-node-transport"),
            "{error}"
        );
    }

    #[test]
    fn a_ca_bundle_must_exist_and_contain_at_least_one_certificate() {
        let missing = NodeTransport::resolve(Some(Path::new("missing-node-ca")), false)
            .expect_err("missing CA is refused");
        assert_eq!(missing.kind(), ErrorKind::NotFound);

        let directory = tempfile::tempdir().expect("temp dir");
        let empty = directory.path().join("empty-node-ca");
        std::fs::write(&empty, b"").expect("writes empty bundle");
        let error =
            NodeTransport::resolve(Some(&empty), false).expect_err("an empty CA bundle is refused");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("no certificates"), "{error}");
    }

    #[test]
    fn a_certificate_bundle_builds_a_server_authentication_config() {
        let certified = rcgen::generate_simple_self_signed(vec!["node.example".to_string()])
            .expect("creates a certificate");
        let bundle = rcgen::Certificate::pem(&certified.cert);
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("node-ca");
        std::fs::write(&path, bundle).expect("writes bundle");

        let transport =
            NodeTransport::resolve(Some(&path), false).expect("valid certificate bundle resolves");
        assert!(transport.tls_config().is_some());
    }
}
