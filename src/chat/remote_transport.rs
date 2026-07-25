//! Outbound, bounded host WebSocket transport for the blind relay.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use camelid_remote_crypto::{RemoteCryptoError, RemoteHandshake, RemoteTransport};
use camelid_remote_store::{RemoteStore, StoreError};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use camelid_remote_protocol::{PairResponse, MAX_NOISE_RECORD_BYTES};

use super::remote_pairing::{PairingCoordinator, PairingError, PendingPairing};

#[derive(Debug, thiserror::Error)]
pub enum RemoteTransportError {
    #[error("relay endpoint is invalid")]
    InvalidEndpoint,
    #[error("relay authentication failed")]
    Unauthorized,
    #[error("relay host is unavailable")]
    Unavailable,
    #[error("relay connection closed")]
    Closed,
    #[error("relay frame is invalid")]
    InvalidFrame,
    #[error("remote peer authentication failed")]
    AuthenticationFailed,
    #[error("remote device is not authorized")]
    UnauthorizedDevice,
    #[error("remote Noise connection does not exist")]
    UnknownConnection,
    #[error("remote pairing failed")]
    PairingFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRelayEndpoint {
    url: String,
}

impl HostRelayEndpoint {
    pub fn new(url: &str) -> Result<Self, RemoteTransportError> {
        let parsed = url::Url::parse(url).map_err(|_| RemoteTransportError::InvalidEndpoint)?;
        let secure = parsed.scheme() == "wss";
        let test_loopback = cfg!(test)
            && parsed.scheme() == "ws"
            && matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "::1"));
        if (!secure && !test_loopback)
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(RemoteTransportError::InvalidEndpoint);
        }
        Ok(Self {
            url: url.trim_end_matches('/').into(),
        })
    }

    fn host_socket_url(&self, route_id: &str) -> String {
        format!("{}/v1/routes/{route_id}/host", self.url)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReconnectPolicy {
    initial: Duration,
    maximum: Duration,
    jitter_percent: u8,
}

impl ReconnectPolicy {
    pub fn new(
        initial: Duration,
        maximum: Duration,
        jitter_percent: u8,
    ) -> Result<Self, RemoteTransportError> {
        if initial.is_zero()
            || maximum < initial
            || jitter_percent > 50
            || maximum.as_millis() > u128::from(u64::MAX)
        {
            return Err(RemoteTransportError::InvalidEndpoint);
        }
        Ok(Self {
            initial,
            maximum,
            jitter_percent,
        })
    }

    pub fn delay(self, failed_attempt: u32, random: u64) -> Duration {
        let exponent = failed_attempt.min(63);
        let base_ms = (self.initial.as_millis() as u64)
            .saturating_mul(1_u64 << exponent)
            .min(self.maximum.as_millis() as u64);
        let jitter_ms = base_ms.saturating_mul(u64::from(self.jitter_percent)) / 100;
        let span = jitter_ms.saturating_mul(2).saturating_add(1);
        let offset = random % span;
        Duration::from_millis(base_ms.saturating_sub(jitter_ms).saturating_add(offset))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedCiphertext {
    pub connection_id: Uuid,
    pub ciphertext: Vec<u8>,
}

pub struct HostRelaySocket {
    socket: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

pub struct RelayEnrollment {
    pub route_id: String,
    pub host_capability: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelayEnrollmentResponse {
    route_id: String,
    host_capability: String,
}

pub struct AuthorizedNoiseSessions {
    host_private: [u8; 32],
    store: Arc<Mutex<RemoteStore>>,
    sessions: HashMap<Uuid, AuthorizedNoiseSession>,
    pending_pairings: HashMap<Uuid, PendingNoisePairing>,
}

struct AuthorizedNoiseSession {
    device_id: Uuid,
    device_noise_public: [u8; 32],
    transport: RemoteTransport,
}

struct PendingNoisePairing {
    device_noise_public: [u8; 32],
    handshake: RemoteHandshake,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedNoiseHandshake {
    pub device_id: Uuid,
    pub initial_payload: Vec<u8>,
    pub response: RoutedCiphertext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedPairingHandshake {
    pub connection_id: Uuid,
    pub device_noise_public: [u8; 32],
    pub handshake_hash: Vec<u8>,
    pub pairing_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedPairing {
    pub device_id: Uuid,
    pub response: RoutedCiphertext,
}

pub fn accept_pairing_with_local_confirmation<F>(
    sessions: &mut AuthorizedNoiseSessions,
    coordinator: &PairingCoordinator,
    frame: RoutedCiphertext,
    session_id: Uuid,
    now_unix_ms: u64,
    confirm: F,
) -> Result<AcceptedPairing, RemoteTransportError>
where
    F: FnOnce(&PendingPairing) -> bool,
{
    let connection_id = frame.connection_id;
    let received = sessions.receive_pairing_first_record(frame)?;
    let pending = match coordinator.receive_authenticated_request(
        &received.pairing_payload,
        connection_id,
        received.device_noise_public,
        &received.handshake_hash,
        now_unix_ms,
    ) {
        Ok(pending) => pending,
        Err(error) => {
            sessions.reject_pairing(connection_id);
            return Err(map_pairing(error));
        }
    };
    let accepted = confirm(&pending);
    finish_pairing_after_confirmation(
        sessions,
        coordinator,
        pending.confirmation_id,
        connection_id,
        &pending.authentication_fingerprint,
        accepted,
        session_id,
        now_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn finish_pairing_after_confirmation(
    sessions: &mut AuthorizedNoiseSessions,
    coordinator: &PairingCoordinator,
    confirmation_id: Uuid,
    connection_id: Uuid,
    authentication_fingerprint: &str,
    accepted: bool,
    session_id: Uuid,
    now_unix_ms: u64,
) -> Result<AcceptedPairing, RemoteTransportError> {
    let paired = match coordinator.confirm(
        confirmation_id,
        connection_id,
        authentication_fingerprint,
        accepted,
        now_unix_ms,
    ) {
        Ok(paired) => paired,
        Err(error) => {
            sessions.reject_pairing(connection_id);
            return Err(map_pairing(error));
        }
    };
    let response = PairResponse {
        v: 1,
        host_id: coordinator.host_id(),
        device_id: paired.device_id,
        session_id,
        supported_capabilities: vec!["agent_events".into(), "session_catalog_v1".into()],
    };
    response
        .validate()
        .map_err(|_| RemoteTransportError::PairingFailed)?;
    let encoded = serde_json::to_vec(&response).map_err(|_| RemoteTransportError::PairingFailed)?;
    let accepted = sessions.finish_pairing(connection_id, &encoded)?;
    if accepted.device_id != paired.device_id {
        sessions.disconnect(connection_id);
        return Err(RemoteTransportError::PairingFailed);
    }
    Ok(AcceptedPairing {
        device_id: paired.device_id,
        response: accepted.response,
    })
}

impl AuthorizedNoiseSessions {
    pub fn new(host_private: &[u8; 32], store: Arc<Mutex<RemoteStore>>) -> Self {
        Self {
            host_private: *host_private,
            store,
            sessions: HashMap::new(),
            pending_pairings: HashMap::new(),
        }
    }

    pub fn accept_first_record(
        &mut self,
        frame: RoutedCiphertext,
    ) -> Result<AcceptedNoiseHandshake, RemoteTransportError> {
        if self.sessions.remove(&frame.connection_id).is_some() {
            return Err(RemoteTransportError::AuthenticationFailed);
        }
        let mut handshake = RemoteHandshake::responder(&self.host_private).map_err(map_crypto)?;
        let initial_payload = handshake.read(&frame.ciphertext).map_err(map_crypto)?;
        let device_key = handshake
            .remote_static()
            .ok_or(RemoteTransportError::AuthenticationFailed)?;
        let device_id = self
            .store
            .lock()
            .map_err(|_| RemoteTransportError::Unavailable)?
            .authorized_device_for_key(&device_key)
            .map_err(map_store)?
            .ok_or(RemoteTransportError::UnauthorizedDevice)?;
        let response_ciphertext = handshake.write(&[]).map_err(map_crypto)?;
        let transport = handshake.into_transport().map_err(map_crypto)?;
        self.sessions.insert(
            frame.connection_id,
            AuthorizedNoiseSession {
                device_id,
                device_noise_public: device_key,
                transport,
            },
        );
        Ok(AcceptedNoiseHandshake {
            device_id,
            initial_payload,
            response: RoutedCiphertext {
                connection_id: frame.connection_id,
                ciphertext: response_ciphertext,
            },
        })
    }

    pub fn receive_pairing_first_record(
        &mut self,
        frame: RoutedCiphertext,
    ) -> Result<ReceivedPairingHandshake, RemoteTransportError> {
        if self.sessions.contains_key(&frame.connection_id)
            || self.pending_pairings.contains_key(&frame.connection_id)
        {
            self.disconnect(frame.connection_id);
            return Err(RemoteTransportError::AuthenticationFailed);
        }
        let mut handshake = RemoteHandshake::responder(&self.host_private).map_err(map_crypto)?;
        let pairing_payload = handshake.read(&frame.ciphertext).map_err(map_crypto)?;
        let device_noise_public = handshake
            .remote_static()
            .ok_or(RemoteTransportError::AuthenticationFailed)?;
        if self
            .store
            .lock()
            .map_err(|_| RemoteTransportError::Unavailable)?
            .authorized_device_for_key(&device_noise_public)
            .map_err(map_store)?
            .is_some()
        {
            return Err(RemoteTransportError::UnauthorizedDevice);
        }
        let handshake_hash = handshake.handshake_hash();
        self.pending_pairings.insert(
            frame.connection_id,
            PendingNoisePairing {
                device_noise_public,
                handshake,
            },
        );
        Ok(ReceivedPairingHandshake {
            connection_id: frame.connection_id,
            device_noise_public,
            handshake_hash,
            pairing_payload,
        })
    }

    pub fn finish_pairing(
        &mut self,
        connection_id: Uuid,
        response_payload: &[u8],
    ) -> Result<AcceptedNoiseHandshake, RemoteTransportError> {
        let mut pending = self
            .pending_pairings
            .remove(&connection_id)
            .ok_or(RemoteTransportError::UnknownConnection)?;
        let device_id = self
            .store
            .lock()
            .map_err(|_| RemoteTransportError::Unavailable)?
            .authorized_device_for_key(&pending.device_noise_public)
            .map_err(map_store)?
            .ok_or(RemoteTransportError::UnauthorizedDevice)?;
        let response_ciphertext = pending
            .handshake
            .write(response_payload)
            .map_err(map_crypto)?;
        let transport = pending.handshake.into_transport().map_err(map_crypto)?;
        self.sessions.insert(
            connection_id,
            AuthorizedNoiseSession {
                device_id,
                device_noise_public: pending.device_noise_public,
                transport,
            },
        );
        Ok(AcceptedNoiseHandshake {
            device_id,
            initial_payload: Vec::new(),
            response: RoutedCiphertext {
                connection_id,
                ciphertext: response_ciphertext,
            },
        })
    }

    pub fn reject_pairing(&mut self, connection_id: Uuid) {
        self.pending_pairings.remove(&connection_id);
    }

    pub fn open(&mut self, frame: RoutedCiphertext) -> Result<Vec<u8>, RemoteTransportError> {
        let result = self
            .sessions
            .get_mut(&frame.connection_id)
            .ok_or(RemoteTransportError::UnknownConnection)?
            .transport
            .open(&frame.ciphertext)
            .map_err(map_crypto);
        if result.is_err() {
            self.sessions.remove(&frame.connection_id);
        }
        result
    }

    pub fn seal(
        &mut self,
        connection_id: Uuid,
        plaintext: &[u8],
    ) -> Result<RoutedCiphertext, RemoteTransportError> {
        let result = self
            .sessions
            .get_mut(&connection_id)
            .ok_or(RemoteTransportError::UnknownConnection)?
            .transport
            .seal(plaintext)
            .map_err(map_crypto);
        match result {
            Ok(ciphertext) => Ok(RoutedCiphertext {
                connection_id,
                ciphertext,
            }),
            Err(error) => {
                self.sessions.remove(&connection_id);
                Err(error)
            }
        }
    }

    pub fn disconnect(&mut self, connection_id: Uuid) {
        self.sessions.remove(&connection_id);
        self.pending_pairings.remove(&connection_id);
    }

    pub fn authenticated_device(&self, connection_id: Uuid) -> Option<(Uuid, [u8; 32])> {
        self.sessions
            .get(&connection_id)
            .map(|session| (session.device_id, session.device_noise_public))
    }

    pub fn reset_for_reconnect(&mut self) {
        self.sessions.clear();
        self.pending_pairings.clear();
    }

    pub fn disconnect_revoked(&mut self) -> Result<Vec<(Uuid, Uuid)>, RemoteTransportError> {
        let revoked = {
            let store = self
                .store
                .lock()
                .map_err(|_| RemoteTransportError::Unavailable)?;
            self.sessions
                .iter()
                .filter_map(|(connection_id, session)| {
                    match store.device_authorized(session.device_id, &session.device_noise_public) {
                        Ok(true) => None,
                        Ok(false) | Err(_) => Some((*connection_id, session.device_id)),
                    }
                })
                .collect::<Vec<_>>()
        };
        for (connection_id, _) in &revoked {
            self.disconnect(*connection_id);
        }
        Ok(revoked)
    }

    pub fn revoke_device(
        &mut self,
        device_id: Uuid,
        now_unix_ms: u64,
    ) -> Result<Vec<Uuid>, RemoteTransportError> {
        self.store
            .lock()
            .map_err(|_| RemoteTransportError::Unavailable)?
            .revoke_device(device_id, now_unix_ms)
            .map_err(map_store)?;
        let connections = self
            .sessions
            .iter()
            .filter_map(|(connection_id, session)| {
                (session.device_id == device_id).then_some(*connection_id)
            })
            .collect::<Vec<_>>();
        self.sessions
            .retain(|_, session| session.device_id != device_id);
        Ok(connections)
    }
}

impl Drop for AuthorizedNoiseSessions {
    fn drop(&mut self) {
        self.host_private.fill(0);
    }
}

impl HostRelaySocket {
    pub async fn connect(
        endpoint: &HostRelayEndpoint,
        route_id: &str,
        host_capability: &str,
    ) -> Result<Self, RemoteTransportError> {
        let mut request = endpoint
            .host_socket_url(route_id)
            .into_client_request()
            .map_err(|_| RemoteTransportError::InvalidEndpoint)?;
        request.headers_mut().insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {host_capability}"))
                .map_err(|_| RemoteTransportError::Unauthorized)?,
        );
        let (socket, _) = connect_async(request).await.map_err(map_connect_error)?;
        Ok(Self { socket })
    }

    pub async fn connect_with_backoff(
        endpoint: &HostRelayEndpoint,
        route_id: &str,
        host_capability: &str,
        policy: ReconnectPolicy,
        cancellation: &CancellationToken,
    ) -> Result<Self, RemoteTransportError> {
        let mut failed_attempt = 0_u32;
        loop {
            match Self::connect(endpoint, route_id, host_capability).await {
                Ok(socket) => return Ok(socket),
                Err(error @ RemoteTransportError::InvalidEndpoint)
                | Err(error @ RemoteTransportError::Unauthorized) => return Err(error),
                Err(_) => {
                    let random = random_u64()?;
                    let delay = policy.delay(failed_attempt, random);
                    failed_attempt = failed_attempt.saturating_add(1);
                    tokio::select! {
                        () = tokio::time::sleep(delay) => {}
                        () = cancellation.cancelled() => {
                            return Err(RemoteTransportError::Closed);
                        }
                    }
                }
            }
        }
    }

    pub async fn receive(&mut self) -> Result<RoutedCiphertext, RemoteTransportError> {
        loop {
            let message = self
                .socket
                .next()
                .await
                .ok_or(RemoteTransportError::Closed)?
                .map_err(|_| RemoteTransportError::Closed)?;
            match message {
                Message::Binary(bytes) => return decode_relay_frame(bytes),
                Message::Ping(_) | Message::Pong(_) => {}
                Message::Close(_) => return Err(RemoteTransportError::Closed),
                Message::Text(_) | Message::Frame(_) => {
                    let _ = self.socket.close(None).await;
                    return Err(RemoteTransportError::InvalidFrame);
                }
            }
        }
    }

    pub async fn send(&mut self, frame: RoutedCiphertext) -> Result<(), RemoteTransportError> {
        validate_ciphertext(&frame.ciphertext)?;
        let mut bytes = Vec::with_capacity(16 + frame.ciphertext.len());
        bytes.extend_from_slice(frame.connection_id.as_bytes());
        bytes.extend_from_slice(&frame.ciphertext);
        self.socket
            .send(Message::Binary(bytes))
            .await
            .map_err(|_| RemoteTransportError::Closed)
    }

    pub async fn disconnect_device(
        &mut self,
        connection_id: Uuid,
    ) -> Result<(), RemoteTransportError> {
        self.socket
            .send(Message::Binary(connection_id.as_bytes().to_vec()))
            .await
            .map_err(|_| RemoteTransportError::Closed)
    }

    pub async fn ping(&mut self) -> Result<(), RemoteTransportError> {
        self.socket
            .send(Message::Ping(Vec::new()))
            .await
            .map_err(|_| RemoteTransportError::Closed)
    }
}

pub async fn enroll_route(
    endpoint: &HostRelayEndpoint,
    enrollment_token: &str,
) -> Result<RelayEnrollment, RemoteTransportError> {
    if !(32..=4096).contains(&enrollment_token.len()) {
        return Err(RemoteTransportError::Unauthorized);
    }
    let response = reqwest::Client::new()
        .post(endpoint.enrollment_url()?)
        .bearer_auth(enrollment_token)
        .send()
        .await
        .map_err(|_| RemoteTransportError::Unavailable)?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err(RemoteTransportError::Unauthorized);
    }
    if response.status() != reqwest::StatusCode::CREATED {
        return Err(RemoteTransportError::Unavailable);
    }
    let response: RelayEnrollmentResponse = response
        .json()
        .await
        .map_err(|_| RemoteTransportError::InvalidFrame)?;
    if !is_canonical_route_id(&response.route_id)
        || !is_canonical_host_capability(&response.host_capability)
    {
        return Err(RemoteTransportError::InvalidFrame);
    }
    Ok(RelayEnrollment {
        route_id: response.route_id,
        host_capability: response.host_capability,
    })
}

impl HostRelayEndpoint {
    fn enrollment_url(&self) -> Result<String, RemoteTransportError> {
        let mut parsed =
            url::Url::parse(&self.url).map_err(|_| RemoteTransportError::InvalidEndpoint)?;
        let scheme = match parsed.scheme() {
            "wss" => "https",
            "ws" if cfg!(test) => "http",
            _ => return Err(RemoteTransportError::InvalidEndpoint),
        };
        parsed
            .set_scheme(scheme)
            .map_err(|_| RemoteTransportError::InvalidEndpoint)?;
        parsed.set_path("/v1/routes/enroll");
        Ok(parsed.into())
    }
}

fn is_canonical_route_id(value: &str) -> bool {
    value.len() == 22
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn is_canonical_host_capability(value: &str) -> bool {
    (32..=4096).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn decode_relay_frame(bytes: Vec<u8>) -> Result<RoutedCiphertext, RemoteTransportError> {
    if bytes.len() <= 16 || bytes.len() > 16 + MAX_NOISE_RECORD_BYTES {
        return Err(RemoteTransportError::InvalidFrame);
    }
    let connection_id = Uuid::from_bytes(
        bytes[..16]
            .try_into()
            .map_err(|_| RemoteTransportError::InvalidFrame)?,
    );
    let ciphertext = bytes[16..].to_vec();
    validate_ciphertext(&ciphertext)?;
    Ok(RoutedCiphertext {
        connection_id,
        ciphertext,
    })
}

fn validate_ciphertext(ciphertext: &[u8]) -> Result<(), RemoteTransportError> {
    if ciphertext.is_empty() || ciphertext.len() > MAX_NOISE_RECORD_BYTES {
        return Err(RemoteTransportError::InvalidFrame);
    }
    Ok(())
}

fn random_u64() -> Result<u64, RemoteTransportError> {
    let mut bytes = [0_u8; 8];
    getrandom::fill(&mut bytes).map_err(|_| RemoteTransportError::Unavailable)?;
    Ok(u64::from_le_bytes(bytes))
}

fn map_connect_error(error: tokio_tungstenite::tungstenite::Error) -> RemoteTransportError {
    match error {
        tokio_tungstenite::tungstenite::Error::Http(response)
            if response.status().as_u16() == 401 || response.status().as_u16() == 404 =>
        {
            RemoteTransportError::Unauthorized
        }
        _ => RemoteTransportError::Unavailable,
    }
}

fn map_crypto(error: RemoteCryptoError) -> RemoteTransportError {
    match error {
        RemoteCryptoError::AuthenticationFailed => RemoteTransportError::AuthenticationFailed,
        RemoteCryptoError::MessageTooLarge => RemoteTransportError::InvalidFrame,
        RemoteCryptoError::Unavailable
        | RemoteCryptoError::InvalidKey
        | RemoteCryptoError::InvalidState => RemoteTransportError::Unavailable,
    }
}

fn map_store(error: StoreError) -> RemoteTransportError {
    match error {
        StoreError::Conflict => RemoteTransportError::UnauthorizedDevice,
        StoreError::Unavailable | StoreError::NewerSchema | StoreError::Invalid => {
            RemoteTransportError::Unavailable
        }
    }
}

fn map_pairing(error: PairingError) -> RemoteTransportError {
    match error {
        PairingError::Unavailable => RemoteTransportError::Unavailable,
        PairingError::Invalid => RemoteTransportError::InvalidFrame,
        PairingError::Expired
        | PairingError::Unauthorized
        | PairingError::ConfirmationMismatch
        | PairingError::Rejected => RemoteTransportError::PairingFailed,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use camelid_relay::server::{app, RelayHttpState};
    use camelid_relay::{RelayRouter, UnavailablePush};
    use camelid_remote_crypto::{RemoteHandshake, StaticKeypair};
    use camelid_remote_store::StoredHostIdentity;
    use serde_json::json;
    use tokio_tungstenite::tungstenite::http::Request;

    use super::*;
    use crate::chat::remote_pairing::PairingCoordinator;

    #[test]
    fn reconnect_policy_is_bounded_and_requires_explicit_parameters() {
        assert!(ReconnectPolicy::new(Duration::ZERO, Duration::from_secs(1), 10).is_err());
        assert!(ReconnectPolicy::new(Duration::from_secs(2), Duration::from_secs(1), 10).is_err());
        let policy =
            ReconnectPolicy::new(Duration::from_millis(100), Duration::from_secs(2), 20).unwrap();
        assert_eq!(policy.delay(0, 0), Duration::from_millis(80));
        assert!(policy.delay(63, u64::MAX) <= Duration::from_millis(2400));
    }

    #[tokio::test]
    async fn host_socket_authenticates_and_routes_without_an_offline_queue() {
        let router = RelayRouter::new(Arc::new(UnavailablePush));
        let route = router.enroll_route().unwrap();
        let state = RelayHttpState::new(router, "test-enrollment-token-that-is-long-enough".into())
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app(state)).await.unwrap() });
        let endpoint = HostRelayEndpoint::new(&format!("ws://{address}")).unwrap();
        assert!(matches!(
            HostRelaySocket::connect(
                &endpoint,
                &route.route_id.expose_for_enrollment(),
                "AAAAAAAAAAAAAAAAAAAAAA"
            )
            .await,
            Err(RemoteTransportError::Unauthorized)
        ));

        let mut host = HostRelaySocket::connect(
            &endpoint,
            &route.route_id.expose_for_enrollment(),
            &route.host_capability.expose_for_enrollment(),
        )
        .await
        .unwrap();
        host.ping().await.unwrap();
        let device_url = format!(
            "ws://{address}/v1/routes/{}/device",
            route.route_id.expose_for_enrollment()
        );
        let mut request = Request::builder()
            .uri(device_url)
            .header("host", address.to_string())
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .body(())
            .unwrap();
        request
            .headers_mut()
            .insert("origin", HeaderValue::from_static("http://127.0.0.1"));
        let (mut device, _) = connect_async(request).await.unwrap();

        let ciphertext = vec![7_u8; 96];
        device
            .send(Message::Binary(ciphertext.clone()))
            .await
            .unwrap();
        let at_host = host.receive().await.unwrap();
        assert_eq!(at_host.ciphertext, ciphertext);
        host.send(at_host.clone()).await.unwrap();
        assert_eq!(
            device.next().await.unwrap().unwrap(),
            Message::Binary(ciphertext)
        );
        host.disconnect_device(at_host.connection_id).await.unwrap();
        let closed = tokio::time::timeout(Duration::from_secs(1), device.next())
            .await
            .expect("relay must close a revoked device promptly");
        assert!(!matches!(closed, Some(Ok(Message::Binary(_)))));
        server.abort();
    }

    #[tokio::test]
    async fn relay_noise_secret_confirmation_and_pair_response_compose_end_to_end() {
        let router = RelayRouter::new(Arc::new(UnavailablePush));
        let route = router.enroll_route().unwrap();
        let state = RelayHttpState::new(router, "test-enrollment-token-that-is-long-enough".into())
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app(state)).await.unwrap() });
        let endpoint = HostRelayEndpoint::new(&format!("ws://{address}")).unwrap();
        let mut host_socket = HostRelaySocket::connect(
            &endpoint,
            &route.route_id.expose_for_enrollment(),
            &route.host_capability.expose_for_enrollment(),
        )
        .await
        .unwrap();
        let request = Request::builder()
            .uri(format!(
                "ws://{address}/v1/connect/{}",
                route.route_id.expose_for_enrollment()
            ))
            .header("host", address.to_string())
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .body(())
            .unwrap();
        let (mut device_socket, _) = connect_async(request).await.unwrap();

        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            RemoteStore::open(&directory.path().join("network-pairing.sqlite3")).unwrap(),
        ));
        let host_key = StaticKeypair::generate().unwrap();
        let host_id = Uuid::new_v4();
        let identity = StoredHostIdentity {
            host_id,
            noise_public: *host_key.public(),
            secret_reference: "os-secret://camelid/host".into(),
        };
        store
            .lock()
            .unwrap()
            .initialize_host_identity(
                identity.host_id,
                &identity.noise_public,
                &identity.secret_reference,
                1,
            )
            .unwrap();
        let coordinator = PairingCoordinator::new(store.clone(), identity);
        let qr = coordinator
            .create_offer(
                "wss://relay.example.test/v1/connect",
                &route.route_id.expose_for_enrollment(),
                10,
            )
            .unwrap();
        let session_id = Uuid::new_v4();
        let device_key = StaticKeypair::generate().unwrap();
        let mut device_noise =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let pair_request = serde_json::to_vec(&json!({
            "pairing_secret": qr.pairing_secret,
            "device_label": "Network phone",
            "app_protocol_version": 1,
            "supported_capabilities": ["agent_events"]
        }))
        .unwrap();
        device_socket
            .send(Message::Binary(device_noise.write(&pair_request).unwrap()))
            .await
            .unwrap();

        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store.clone());
        let frame = host_socket.receive().await.unwrap();
        let connection_id = frame.connection_id;
        let accepted = accept_pairing_with_local_confirmation(
            &mut sessions,
            &coordinator,
            frame,
            session_id,
            11,
            |pending| {
                assert_eq!(pending.device_label, "Network phone");
                !pending.authentication_fingerprint.is_empty()
            },
        )
        .unwrap();
        host_socket.send(accepted.response).await.unwrap();

        let Message::Binary(response_record) = device_socket.next().await.unwrap().unwrap() else {
            panic!("device expected binary pairing response")
        };
        let response = device_noise.read(&response_record).unwrap();
        let response = PairResponse::decode(&response).unwrap();
        assert_eq!(response.host_id, host_id);
        assert_eq!(response.device_id, accepted.device_id);
        assert_eq!(response.session_id, session_id);
        assert!(store
            .lock()
            .unwrap()
            .device_authorized(response.device_id, device_key.public())
            .unwrap());

        let mut device_transport = device_noise.into_transport().unwrap();
        device_socket
            .send(Message::Binary(
                device_transport
                    .seal(b"authenticated replay request")
                    .unwrap(),
            ))
            .await
            .unwrap();
        let frame = host_socket.receive().await.unwrap();
        assert_eq!(frame.connection_id, connection_id);
        assert_eq!(
            sessions.open(frame).unwrap(),
            b"authenticated replay request"
        );
        server.abort();
    }

    #[tokio::test]
    async fn stolen_route_capability_without_device_key_cannot_reach_commands() {
        let router = RelayRouter::new(Arc::new(UnavailablePush));
        let route = router.enroll_route().unwrap();
        let state = RelayHttpState::new(router, "test-enrollment-token-that-is-long-enough".into())
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app(state)).await.unwrap() });
        let endpoint = HostRelayEndpoint::new(&format!("ws://{address}")).unwrap();
        let mut host = HostRelaySocket::connect(
            &endpoint,
            &route.route_id.expose_for_enrollment(),
            &route.host_capability.expose_for_enrollment(),
        )
        .await
        .unwrap();
        let request = Request::builder()
            .uri(format!(
                "ws://{address}/v1/routes/{}/device",
                route.route_id.expose_for_enrollment()
            ))
            .header("host", address.to_string())
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .body(())
            .unwrap();
        let (mut attacker, _) = connect_async(request).await.unwrap();

        let host_key = StaticKeypair::generate().unwrap();
        let attacker_key = StaticKeypair::generate().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            RemoteStore::open(&directory.path().join("stolen-route.sqlite3")).unwrap(),
        ));
        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store);
        let mut initiator =
            RemoteHandshake::initiator(attacker_key.private(), host_key.public()).unwrap();
        attacker
            .send(Message::Binary(
                initiator.write(b"forged start_turn command").unwrap(),
            ))
            .await
            .unwrap();
        let frame = host.receive().await.unwrap();
        let connection_id = frame.connection_id;
        assert!(matches!(
            sessions.accept_first_record(frame),
            Err(RemoteTransportError::UnauthorizedDevice)
        ));
        assert!(matches!(
            sessions.seal(connection_id, b"no command result"),
            Err(RemoteTransportError::UnknownConnection)
        ));
        host.disconnect_device(connection_id).await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn host_connector_retries_until_relay_exists_and_honors_cancellation() {
        let reserved = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = reserved.local_addr().unwrap();
        drop(reserved);
        let router = RelayRouter::new(Arc::new(UnavailablePush));
        let route = router.enroll_route().unwrap();
        let state = RelayHttpState::new(router, "test-enrollment-token-that-is-long-enough".into())
            .unwrap();
        let endpoint = HostRelayEndpoint::new(&format!("ws://{address}")).unwrap();
        let route_id = route.route_id.expose_for_enrollment();
        let capability = route.host_capability.expose_for_enrollment();
        let policy =
            ReconnectPolicy::new(Duration::from_millis(5), Duration::from_millis(20), 0).unwrap();
        let cancellation = CancellationToken::new();
        let connector = {
            let endpoint = endpoint.clone();
            let cancellation = cancellation.clone();
            let route_id = route_id.clone();
            let capability = capability.clone();
            tokio::spawn(async move {
                HostRelaySocket::connect_with_backoff(
                    &endpoint,
                    &route_id,
                    &capability,
                    policy,
                    &cancellation,
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(15)).await;
        let listener = tokio::net::TcpListener::bind(address).await.unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app(state)).await.unwrap() });
        let _socket = tokio::time::timeout(Duration::from_secs(1), connector)
            .await
            .expect("connector must observe relay recovery")
            .unwrap()
            .unwrap();
        server.abort();

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            HostRelaySocket::connect_with_backoff(
                &endpoint,
                &route_id,
                &capability,
                policy,
                &cancellation,
            )
            .await,
            Err(RemoteTransportError::Closed)
        ));
    }

    #[test]
    fn noise_sessions_authorize_tamper_terminally_and_revoke_live_connections() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("noise.sqlite3")).unwrap();
        let host_key = StaticKeypair::generate().unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let device_id = Uuid::new_v4();
        store
            .register_device(device_id, "Phone", device_key.public(), 1)
            .unwrap();
        let mut sessions =
            AuthorizedNoiseSessions::new(host_key.private(), Arc::new(Mutex::new(store)));
        let connection_id = Uuid::new_v4();
        let mut device =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let accepted = sessions
            .accept_first_record(RoutedCiphertext {
                connection_id,
                ciphertext: device.write(b"replay-from-42").unwrap(),
            })
            .unwrap();
        assert_eq!(accepted.device_id, device_id);
        assert_eq!(accepted.initial_payload, b"replay-from-42");
        device.read(&accepted.response.ciphertext).unwrap();
        let mut device = device.into_transport().unwrap();

        let inbound = device.seal(b"authenticated command").unwrap();
        assert_eq!(
            sessions
                .open(RoutedCiphertext {
                    connection_id,
                    ciphertext: inbound,
                })
                .unwrap(),
            b"authenticated command"
        );
        let outbound = sessions.seal(connection_id, b"committed event").unwrap();
        assert_eq!(
            device.open(&outbound.ciphertext).unwrap(),
            b"committed event"
        );

        let mut tampered = device.seal(b"tamper me").unwrap();
        tampered[0] ^= 1;
        assert!(matches!(
            sessions.open(RoutedCiphertext {
                connection_id,
                ciphertext: tampered,
            }),
            Err(RemoteTransportError::AuthenticationFailed)
        ));
        assert!(matches!(
            sessions.seal(connection_id, b"must be gone"),
            Err(RemoteTransportError::UnknownConnection)
        ));

        let second_connection = Uuid::new_v4();
        let mut reconnect =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let accepted = sessions
            .accept_first_record(RoutedCiphertext {
                connection_id: second_connection,
                ciphertext: reconnect.write(&[]).unwrap(),
            })
            .unwrap();
        reconnect.read(&accepted.response.ciphertext).unwrap();
        sessions.reset_for_reconnect();
        assert!(matches!(
            sessions.seal(second_connection, b"disconnected"),
            Err(RemoteTransportError::UnknownConnection)
        ));

        let third_connection = Uuid::new_v4();
        let mut reconnect =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let accepted = sessions
            .accept_first_record(RoutedCiphertext {
                connection_id: third_connection,
                ciphertext: reconnect.write(&[]).unwrap(),
            })
            .unwrap();
        reconnect.read(&accepted.response.ciphertext).unwrap();
        assert_eq!(
            sessions.revoke_device(device_id, 2).unwrap(),
            [third_connection]
        );

        let mut after_revoke =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        assert!(matches!(
            sessions.accept_first_record(RoutedCiphertext {
                connection_id: Uuid::new_v4(),
                ciphertext: after_revoke.write(&[]).unwrap(),
            }),
            Err(RemoteTransportError::UnauthorizedDevice)
        ));
    }

    #[test]
    fn wrong_host_key_and_unregistered_device_cannot_create_noise_sessions() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            RemoteStore::open(&directory.path().join("unauthorized.sqlite3")).unwrap(),
        ));
        let host_key = StaticKeypair::generate().unwrap();
        let wrong_host = StaticKeypair::generate().unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store);

        let mut wrong =
            RemoteHandshake::initiator(device_key.private(), wrong_host.public()).unwrap();
        assert!(matches!(
            sessions.accept_first_record(RoutedCiphertext {
                connection_id: Uuid::new_v4(),
                ciphertext: wrong.write(&[]).unwrap(),
            }),
            Err(RemoteTransportError::AuthenticationFailed)
        ));

        let mut unregistered =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        assert!(matches!(
            sessions.accept_first_record(RoutedCiphertext {
                connection_id: Uuid::new_v4(),
                ciphertext: unregistered.write(&[]).unwrap(),
            }),
            Err(RemoteTransportError::UnauthorizedDevice)
        ));
    }

    #[test]
    fn external_store_revocation_prunes_live_noise_session() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("external-revoke.sqlite3");
        let host_key = StaticKeypair::generate().unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let device_id = Uuid::new_v4();
        let mut store = RemoteStore::open(&path).unwrap();
        store
            .register_device(device_id, "Phone", device_key.public(), 1)
            .unwrap();
        let store = Arc::new(Mutex::new(store));
        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store);
        let connection_id = Uuid::new_v4();
        let mut device =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let accepted = sessions
            .accept_first_record(RoutedCiphertext {
                connection_id,
                ciphertext: device.write(&[]).unwrap(),
            })
            .unwrap();
        device.read(&accepted.response.ciphertext).unwrap();

        RemoteStore::open(&path)
            .unwrap()
            .revoke_device(device_id, 2)
            .unwrap();
        assert_eq!(
            sessions.disconnect_revoked().unwrap(),
            vec![(connection_id, device_id)]
        );
        assert!(matches!(
            sessions.seal(connection_id, b"must be closed"),
            Err(RemoteTransportError::UnknownConnection)
        ));
    }

    #[test]
    fn encrypted_pairing_promotes_only_after_matching_local_confirmation() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("pairing-e2e.sqlite3")).unwrap();
        let host_key = StaticKeypair::generate().unwrap();
        let host = StoredHostIdentity {
            host_id: Uuid::new_v4(),
            noise_public: *host_key.public(),
            secret_reference: "os-secret://camelid/host".into(),
        };
        store
            .initialize_host_identity(host.host_id, &host.noise_public, &host.secret_reference, 1)
            .unwrap();
        let store = Arc::new(Mutex::new(store));
        let pairing = PairingCoordinator::new(store.clone(), host);
        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store);
        let qr = pairing
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let request = serde_json::to_vec(&json!({
            "pairing_secret": qr.pairing_secret,
            "device_label": "Phone",
            "app_protocol_version": 1,
            "supported_capabilities": ["agent_events"]
        }))
        .unwrap();
        let connection_id = Uuid::new_v4();
        let mut device =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let received = sessions
            .receive_pairing_first_record(RoutedCiphertext {
                connection_id,
                ciphertext: device.write(&request).unwrap(),
            })
            .unwrap();
        assert!(matches!(
            sessions.seal(connection_id, b"not authorized yet"),
            Err(RemoteTransportError::UnknownConnection)
        ));
        let pending = pairing
            .receive_authenticated_request(
                &received.pairing_payload,
                connection_id,
                received.device_noise_public,
                &received.handshake_hash,
                11,
            )
            .unwrap();
        let paired = pairing
            .confirm(
                pending.confirmation_id,
                connection_id,
                &pending.authentication_fingerprint,
                true,
                12,
            )
            .unwrap();
        let accepted = sessions
            .finish_pairing(connection_id, br#"{"status":"paired"}"#)
            .unwrap();
        assert_eq!(accepted.device_id, paired.device_id);
        assert_eq!(
            device.read(&accepted.response.ciphertext).unwrap(),
            br#"{"status":"paired"}"#
        );
        let mut device = device.into_transport().unwrap();
        let event = sessions.seal(connection_id, b"committed event").unwrap();
        assert_eq!(device.open(&event.ciphertext).unwrap(), b"committed event");
    }

    #[test]
    fn rejected_local_pairing_destroys_pending_noise_state() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            RemoteStore::open(&directory.path().join("pairing-reject.sqlite3")).unwrap(),
        ));
        let host_key = StaticKeypair::generate().unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let mut sessions = AuthorizedNoiseSessions::new(host_key.private(), store);
        let mut device =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let connection_id = Uuid::new_v4();
        sessions
            .receive_pairing_first_record(RoutedCiphertext {
                connection_id,
                ciphertext: device.write(b"pair request").unwrap(),
            })
            .unwrap();
        sessions.reject_pairing(connection_id);
        assert!(matches!(
            sessions.finish_pairing(connection_id, &[]),
            Err(RemoteTransportError::UnknownConnection)
        ));
    }
}
