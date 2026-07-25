//! Blind, bounded routing core for Camelid remote control.
//!
//! This crate has no dependency on Camelid chat, tools, inference, SQLite, or
//! the decrypted application protocol. It routes opaque Noise records by
//! unguessable capabilities and connection IDs. An offline host means refusal,
//! never delayed command storage.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use tokio::sync::mpsc::{channel, error::TrySendError, Receiver, Sender};
use uuid::Uuid;

pub mod server;

pub const MAX_FRAME_BYTES: usize = 65_535;
pub const MAX_DEVICES_PER_ROUTE: usize = 4;
pub const FRAME_QUEUE_CAPACITY: usize = 32;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RelayError {
    #[error("relay route is unavailable")]
    RouteUnavailable,
    #[error("relay capability is invalid")]
    Unauthorized,
    #[error("relay host is already connected")]
    HostAlreadyConnected,
    #[error("relay host is offline")]
    HostOffline,
    #[error("relay device limit reached")]
    DeviceLimit,
    #[error("relay frame is invalid")]
    InvalidFrame,
    #[error("relay connection is backpressured")]
    Backpressure,
    #[error("relay connection is closed")]
    Closed,
    #[error("relay push provider is unavailable")]
    PushUnavailable,
    #[error("relay route persistence is unavailable")]
    PersistenceUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RouteId(Uuid);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HostCapability(Uuid);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NotificationCapability(Uuid);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnectionId(Uuid);

#[derive(Clone, Copy)]
pub struct EnrolledRoute {
    pub route_id: RouteId,
    pub host_capability: HostCapability,
}

#[derive(Clone, PartialEq, Eq)]
pub struct OpaqueFrame {
    pub connection_id: ConnectionId,
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushCategory {
    ApprovalRequired,
    TurnFinished,
    HostAttention,
}

pub trait PushProvider: Send + Sync {
    fn send(&self, platform_token: &str, category: PushCategory) -> Result<(), RelayError>;
}

struct Route {
    host_capability: HostCapability,
    host_sender: Option<Sender<OpaqueFrame>>,
    devices: HashMap<ConnectionId, Sender<Vec<u8>>>,
}

struct State {
    routes: HashMap<RouteId, Route>,
    notifications: HashMap<NotificationCapability, String>,
}

#[derive(Clone)]
pub struct RelayRouter {
    state: Arc<Mutex<State>>,
    push: Arc<dyn PushProvider>,
}

impl RelayRouter {
    pub fn new(push: Arc<dyn PushProvider>) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                routes: HashMap::new(),
                notifications: HashMap::new(),
            })),
            push,
        }
    }

    pub fn enroll_route(&self) -> Result<EnrolledRoute, RelayError> {
        let mut state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route_id = RouteId(Uuid::new_v4());
        let host_capability = HostCapability(Uuid::new_v4());
        state.routes.insert(
            route_id,
            Route {
                host_capability,
                host_sender: None,
                devices: HashMap::new(),
            },
        );
        Ok(EnrolledRoute {
            route_id,
            host_capability,
        })
    }

    pub fn restore_route(&self, route_id: &str, host_capability: &str) -> Result<(), RelayError> {
        let route_id = RouteId::parse(route_id)?;
        let host_capability = HostCapability::parse(host_capability)?;
        let mut state = self.state.lock().map_err(|_| RelayError::Closed)?;
        if state.routes.contains_key(&route_id) {
            return Err(RelayError::Unauthorized);
        }
        state.routes.insert(
            route_id,
            Route {
                host_capability,
                host_sender: None,
                devices: HashMap::new(),
            },
        );
        Ok(())
    }

    pub fn remove_route(&self, route_id: RouteId) -> Result<(), RelayError> {
        self.state
            .lock()
            .map_err(|_| RelayError::Closed)?
            .routes
            .remove(&route_id)
            .map(|_| ())
            .ok_or(RelayError::RouteUnavailable)
    }

    pub fn connect_host(
        &self,
        route_id: RouteId,
        capability: HostCapability,
    ) -> Result<HostConnection, RelayError> {
        let (sender, receiver) = channel(FRAME_QUEUE_CAPACITY);
        let mut state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route = state
            .routes
            .get_mut(&route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        if route.host_capability != capability {
            return Err(RelayError::Unauthorized);
        }
        if route.host_sender.is_some() {
            return Err(RelayError::HostAlreadyConnected);
        }
        route.host_sender = Some(sender);
        Ok(HostConnection {
            route_id,
            receiver,
            state: Arc::clone(&self.state),
        })
    }

    pub fn connect_device(&self, route_id: RouteId) -> Result<DeviceConnection, RelayError> {
        let (sender, receiver) = channel(FRAME_QUEUE_CAPACITY);
        let mut state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route = state
            .routes
            .get_mut(&route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        if route.host_sender.is_none() {
            return Err(RelayError::HostOffline);
        }
        if route.devices.len() >= MAX_DEVICES_PER_ROUTE {
            return Err(RelayError::DeviceLimit);
        }
        let connection_id = ConnectionId(Uuid::new_v4());
        route.devices.insert(connection_id, sender);
        Ok(DeviceConnection {
            route_id,
            connection_id,
            receiver,
            state: Arc::clone(&self.state),
        })
    }

    pub fn register_notification(
        &self,
        platform_token: &str,
    ) -> Result<NotificationCapability, RelayError> {
        if platform_token.trim().is_empty() || platform_token.len() > 4096 {
            return Err(RelayError::Unauthorized);
        }
        let capability = NotificationCapability(Uuid::new_v4());
        self.state
            .lock()
            .map_err(|_| RelayError::Closed)?
            .notifications
            .insert(capability, platform_token.to_string());
        Ok(capability)
    }

    pub fn authorize_device(&self, route_id: RouteId) -> Result<(), RelayError> {
        let state = self.state.lock().map_err(|_| RelayError::Closed)?;
        state
            .routes
            .get(&route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        Ok(())
    }

    pub fn revoke_notification(
        &self,
        capability: NotificationCapability,
    ) -> Result<(), RelayError> {
        if self
            .state
            .lock()
            .map_err(|_| RelayError::Closed)?
            .notifications
            .remove(&capability)
            .is_none()
        {
            return Err(RelayError::Unauthorized);
        }
        Ok(())
    }

    pub fn notify(
        &self,
        capability: NotificationCapability,
        category: PushCategory,
    ) -> Result<(), RelayError> {
        let token = self
            .state
            .lock()
            .map_err(|_| RelayError::Closed)?
            .notifications
            .get(&capability)
            .cloned()
            .ok_or(RelayError::Unauthorized)?;
        self.push.send(&token, category)
    }
}

pub struct HostConnection {
    route_id: RouteId,
    receiver: Receiver<OpaqueFrame>,
    state: Arc<Mutex<State>>,
}

impl HostConnection {
    pub async fn receive(&mut self) -> Result<OpaqueFrame, RelayError> {
        self.receiver.recv().await.ok_or(RelayError::Closed)
    }

    pub fn send(&self, frame: OpaqueFrame) -> Result<(), RelayError> {
        self.sender().send(frame)
    }

    pub fn sender(&self) -> HostSender {
        HostSender {
            route_id: self.route_id,
            state: Arc::clone(&self.state),
        }
    }
}

#[derive(Clone)]
pub struct HostSender {
    route_id: RouteId,
    state: Arc<Mutex<State>>,
}

impl HostSender {
    pub fn send(&self, frame: OpaqueFrame) -> Result<(), RelayError> {
        validate_payload(&frame.payload)?;
        let state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route = state
            .routes
            .get(&self.route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        let sender = route
            .devices
            .get(&frame.connection_id)
            .ok_or(RelayError::Closed)?;
        send_bounded(sender, frame.payload)
    }

    pub fn disconnect(&self, connection_id: ConnectionId) -> Result<(), RelayError> {
        let mut state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route = state
            .routes
            .get_mut(&self.route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        route
            .devices
            .remove(&connection_id)
            .map(|_| ())
            .ok_or(RelayError::Closed)
    }
}

impl Drop for HostConnection {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(route) = state.routes.get_mut(&self.route_id) {
                route.host_sender = None;
                route.devices.clear();
            }
        }
    }
}

pub struct DeviceConnection {
    route_id: RouteId,
    connection_id: ConnectionId,
    receiver: Receiver<Vec<u8>>,
    state: Arc<Mutex<State>>,
}

impl DeviceConnection {
    pub fn connection_id(&self) -> ConnectionId {
        self.connection_id
    }

    pub fn send(&self, payload: Vec<u8>) -> Result<(), RelayError> {
        self.sender().send(payload)
    }

    pub fn sender(&self) -> DeviceSender {
        DeviceSender {
            route_id: self.route_id,
            connection_id: self.connection_id,
            state: Arc::clone(&self.state),
        }
    }

    pub async fn receive(&mut self) -> Result<Vec<u8>, RelayError> {
        self.receiver.recv().await.ok_or(RelayError::Closed)
    }
}

#[derive(Clone)]
pub struct DeviceSender {
    route_id: RouteId,
    connection_id: ConnectionId,
    state: Arc<Mutex<State>>,
}

impl DeviceSender {
    pub fn send(&self, payload: Vec<u8>) -> Result<(), RelayError> {
        validate_payload(&payload)?;
        let state = self.state.lock().map_err(|_| RelayError::Closed)?;
        let route = state
            .routes
            .get(&self.route_id)
            .ok_or(RelayError::RouteUnavailable)?;
        let sender = route.host_sender.as_ref().ok_or(RelayError::HostOffline)?;
        send_bounded(
            sender,
            OpaqueFrame {
                connection_id: self.connection_id,
                payload,
            },
        )
    }
}

impl ConnectionId {
    pub fn to_bytes(self) -> [u8; 16] {
        *self.0.as_bytes()
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Uuid::from_bytes(bytes))
    }
}

impl Drop for DeviceConnection {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(route) = state.routes.get_mut(&self.route_id) {
                route.devices.remove(&self.connection_id);
            }
        }
    }
}

fn validate_payload(payload: &[u8]) -> Result<(), RelayError> {
    if payload.is_empty() || payload.len() > MAX_FRAME_BYTES {
        return Err(RelayError::InvalidFrame);
    }
    Ok(())
}

fn send_bounded<T>(sender: &Sender<T>, value: T) -> Result<(), RelayError> {
    sender.try_send(value).map_err(|error| match error {
        TrySendError::Full(_) => RelayError::Backpressure,
        TrySendError::Closed(_) => RelayError::Closed,
    })
}

pub struct UnavailablePush;

impl PushProvider for UnavailablePush {
    fn send(&self, _platform_token: &str, _category: PushCategory) -> Result<(), RelayError> {
        Err(RelayError::PushUnavailable)
    }
}

impl RouteId {
    pub fn parse(value: &str) -> Result<Self, RelayError> {
        let bytes = URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| RelayError::RouteUnavailable)?;
        let bytes: [u8; 16] = bytes.try_into().map_err(|_| RelayError::RouteUnavailable)?;
        Ok(Self(Uuid::from_bytes(bytes)))
    }

    pub fn expose_for_enrollment(self) -> String {
        URL_SAFE_NO_PAD.encode(self.0.as_bytes())
    }
}

impl HostCapability {
    pub fn parse(value: &str) -> Result<Self, RelayError> {
        Uuid::parse_str(value)
            .map(Self)
            .map_err(|_| RelayError::Unauthorized)
    }

    pub fn expose_for_enrollment(self) -> String {
        self.0.to_string()
    }
}

impl NotificationCapability {
    pub fn parse(value: &str) -> Result<Self, RelayError> {
        Uuid::parse_str(value)
            .map(Self)
            .map_err(|_| RelayError::Unauthorized)
    }

    pub fn expose_once(self) -> String {
        self.0.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use camelid_remote_crypto::{RemoteHandshake, StaticKeypair};

    #[derive(Default)]
    struct FakePush(Mutex<Vec<(String, PushCategory)>>);

    impl PushProvider for FakePush {
        fn send(&self, token: &str, category: PushCategory) -> Result<(), RelayError> {
            self.0
                .lock()
                .map_err(|_| RelayError::Closed)?
                .push((token.into(), category));
            Ok(())
        }
    }

    fn router() -> (RelayRouter, Arc<FakePush>) {
        let push = Arc::new(FakePush::default());
        (RelayRouter::new(push.clone()), push)
    }

    #[tokio::test]
    async fn host_offline_refuses_devices_without_queueing() {
        let (relay, _) = router();
        let route = relay.enroll_route().unwrap();
        assert!(matches!(
            relay.connect_device(route.route_id),
            Err(RelayError::HostOffline)
        ));
    }

    #[tokio::test]
    async fn separate_capabilities_and_connection_limits_are_enforced() {
        let (relay, _) = router();
        let route = relay.enroll_route().unwrap();
        assert!(matches!(
            relay.connect_host(route.route_id, HostCapability(Uuid::new_v4())),
            Err(RelayError::Unauthorized)
        ));
        let _host = relay
            .connect_host(route.route_id, route.host_capability)
            .unwrap();
        assert!(matches!(
            relay.connect_device(RouteId(Uuid::new_v4())),
            Err(RelayError::RouteUnavailable)
        ));
        let devices = (0..MAX_DEVICES_PER_ROUTE)
            .map(|_| relay.connect_device(route.route_id).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(devices.len(), MAX_DEVICES_PER_ROUTE);
        assert!(matches!(
            relay.connect_device(route.route_id),
            Err(RelayError::DeviceLimit)
        ));
    }

    #[tokio::test]
    async fn opaque_noise_records_round_trip_without_plaintext_at_the_relay() {
        let (relay, _) = router();
        let route = relay.enroll_route().unwrap();
        let mut host_connection = relay
            .connect_host(route.route_id, route.host_capability)
            .unwrap();
        let device_connection = relay.connect_device(route.route_id).unwrap();
        let host_key = StaticKeypair::generate().unwrap();
        let device_key = StaticKeypair::generate().unwrap();
        let mut initiator =
            RemoteHandshake::initiator(device_key.private(), host_key.public()).unwrap();
        let first = initiator.write(b"sensitive-pairing-secret").unwrap();
        assert!(!first
            .windows(b"sensitive-pairing-secret".len())
            .any(|window| window == b"sensitive-pairing-secret"));
        device_connection.send(first.clone()).unwrap();
        let at_host = host_connection.receive().await.unwrap();
        assert_eq!(at_host.payload, first);
        assert_eq!(at_host.connection_id, device_connection.connection_id());
    }

    #[tokio::test]
    async fn oversized_and_slow_consumer_frames_fail_bounded() {
        let (relay, _) = router();
        let route = relay.enroll_route().unwrap();
        let _host = relay
            .connect_host(route.route_id, route.host_capability)
            .unwrap();
        let device = relay.connect_device(route.route_id).unwrap();
        assert_eq!(
            device.send(vec![0; MAX_FRAME_BYTES + 1]),
            Err(RelayError::InvalidFrame)
        );
        for _ in 0..FRAME_QUEUE_CAPACITY {
            device.send(vec![1]).unwrap();
        }
        assert_eq!(device.send(vec![1]), Err(RelayError::Backpressure));
    }

    #[tokio::test]
    async fn push_accepts_only_fixed_category_and_capability() {
        let (relay, push) = router();
        let capability = relay.register_notification("platform-token").unwrap();
        relay
            .notify(capability, PushCategory::ApprovalRequired)
            .unwrap();
        assert!(matches!(
            relay.notify(
                NotificationCapability(Uuid::new_v4()),
                PushCategory::HostAttention
            ),
            Err(RelayError::Unauthorized)
        ));
        assert_eq!(
            *push.0.lock().unwrap(),
            [("platform-token".into(), PushCategory::ApprovalRequired)]
        );
    }

    #[tokio::test]
    async fn host_disconnect_closes_exactly_one_open_device() {
        let (relay, _) = router();
        let route = relay.enroll_route().unwrap();
        let host = relay
            .connect_host(route.route_id, route.host_capability)
            .unwrap();
        let mut first = relay.connect_device(route.route_id).unwrap();
        let second = relay.connect_device(route.route_id).unwrap();
        host.sender().disconnect(first.connection_id()).unwrap();
        assert_eq!(first.receive().await, Err(RelayError::Closed));
        second.send(vec![1]).unwrap();
    }
}
