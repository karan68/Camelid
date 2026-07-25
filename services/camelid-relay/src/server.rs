use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};

use crate::{
    ConnectionId, DeviceConnection, EnrolledRoute, HostCapability, HostConnection,
    NotificationCapability, OpaqueFrame, PushCategory, RelayError, RelayRouter, RouteId,
};

pub trait RoutePersistence: Send + Sync {
    fn insert(&self, route_id: &str, host_capability: &str) -> Result<(), RelayError>;
}

struct EphemeralRoutes;

impl RoutePersistence for EphemeralRoutes {
    fn insert(&self, _: &str, _: &str) -> Result<(), RelayError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct RelayHttpState {
    pub router: RelayRouter,
    enrollment_token: Arc<str>,
    route_persistence: Arc<dyn RoutePersistence>,
    keepalive_interval: Option<Duration>,
}

impl RelayHttpState {
    pub fn new(router: RelayRouter, enrollment_token: String) -> Result<Self, RelayError> {
        if enrollment_token.len() < 32 || enrollment_token.len() > 4096 {
            return Err(RelayError::Unauthorized);
        }
        Ok(Self {
            router,
            enrollment_token: enrollment_token.into(),
            route_persistence: Arc::new(EphemeralRoutes),
            keepalive_interval: None,
        })
    }

    pub fn with_route_persistence(mut self, persistence: Arc<dyn RoutePersistence>) -> Self {
        self.route_persistence = persistence;
        self
    }

    pub fn with_keepalive_interval(mut self, interval: Duration) -> Self {
        debug_assert!(!interval.is_zero());
        self.keepalive_interval = Some(interval);
        self
    }
}

pub fn app(state: RelayHttpState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/routes/enroll", post(enroll))
        .route("/v1/routes/:route_id/host", get(host_socket))
        .route("/v1/routes/:route_id/device", get(device_socket))
        .route("/v1/connect/:route_id", get(device_socket))
        .route("/v1/push/register", post(register_push))
        .route("/v1/push/notify", post(notify_push))
        .route("/v1/push/:capability", delete(revoke_push))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

#[derive(Serialize)]
struct EnrollResponse {
    route_id: String,
    host_capability: String,
}

impl From<EnrolledRoute> for EnrollResponse {
    fn from(route: EnrolledRoute) -> Self {
        Self {
            route_id: route.route_id.expose_for_enrollment(),
            host_capability: route.host_capability.expose_for_enrollment(),
        }
    }
}

async fn enroll(
    State(state): State<RelayHttpState>,
    headers: HeaderMap,
) -> Result<(StatusCode, Json<EnrollResponse>), StatusCode> {
    let presented = bearer(&headers).ok_or(StatusCode::UNAUTHORIZED)?;
    if !constant_time_equal(presented.as_bytes(), state.enrollment_token.as_bytes()) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let route = state.router.enroll_route().map_err(map_http_error)?;
    let response = EnrollResponse::from(route);
    if state
        .route_persistence
        .insert(&response.route_id, &response.host_capability)
        .is_err()
    {
        let _ = state.router.remove_route(route.route_id);
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    Ok((StatusCode::CREATED, Json(response)))
}

async fn host_socket(
    State(state): State<RelayHttpState>,
    Path(route_id): Path<String>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let keepalive_interval = state.keepalive_interval;
    let connection = bearer(&headers)
        .ok_or(RelayError::Unauthorized)
        .and_then(HostCapability::parse)
        .and_then(|capability| RouteId::parse(&route_id).map(|route| (route, capability)))
        .and_then(|(route, capability)| state.router.connect_host(route, capability));
    match connection {
        Ok(connection) => upgrade
            .max_message_size(crate::MAX_FRAME_BYTES + 16)
            .on_upgrade(move |socket| bridge_host(socket, connection, keepalive_interval)),
        Err(error) => map_http_error(error).into_response(),
    }
}

async fn device_socket(
    State(state): State<RelayHttpState>,
    Path(route_id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let keepalive_interval = state.keepalive_interval;
    let connection = RouteId::parse(&route_id).and_then(|route| state.router.connect_device(route));
    match connection {
        Ok(connection) => upgrade
            .max_message_size(crate::MAX_FRAME_BYTES)
            .on_upgrade(move |socket| bridge_device(socket, connection, keepalive_interval)),
        Err(error) => map_http_error(error).into_response(),
    }
}

async fn bridge_host(
    socket: WebSocket,
    mut connection: HostConnection,
    keepalive_interval: Option<Duration>,
) {
    let sender = connection.sender();
    let (mut outgoing, mut incoming) = socket.split();
    let mut keepalive = keepalive_timer(keepalive_interval);
    loop {
        tokio::select! {
            _ = next_keepalive(&mut keepalive) => {
                if outgoing.send(Message::Ping(Vec::new())).await.is_err() { break; }
            }
            from_device = connection.receive() => {
                let Ok(frame) = from_device else { break };
                let mut encoded = Vec::with_capacity(16 + frame.payload.len());
                encoded.extend_from_slice(&frame.connection_id.to_bytes());
                encoded.extend_from_slice(&frame.payload);
                if outgoing.send(Message::Binary(encoded)).await.is_err() { break; }
            }
            from_host = incoming.next() => {
                let Some(Ok(message)) = from_host else { break };
                match message {
                    Message::Binary(bytes) => {
                        let Some(frame) = decode_host_frame(bytes) else { break };
                        let result = match frame {
                            HostFrame::Ciphertext { connection_id, payload } => {
                                sender.send(OpaqueFrame { connection_id, payload })
                            }
                            HostFrame::Disconnect(connection_id) => sender.disconnect(connection_id),
                        };
                        if result.is_err() { break; }
                    }
                    Message::Ping(payload) => {
                        if outgoing.send(Message::Pong(payload)).await.is_err() { break; }
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) | Message::Text(_) => break,
                }
            }
        }
    }
}

async fn bridge_device(
    socket: WebSocket,
    mut connection: DeviceConnection,
    keepalive_interval: Option<Duration>,
) {
    let sender = connection.sender();
    let (mut outgoing, mut incoming) = socket.split();
    let mut keepalive = keepalive_timer(keepalive_interval);
    loop {
        tokio::select! {
            _ = next_keepalive(&mut keepalive) => {
                if outgoing.send(Message::Ping(Vec::new())).await.is_err() { break; }
            }
            from_host = connection.receive() => {
                let Ok(payload) = from_host else { break };
                if outgoing.send(Message::Binary(payload)).await.is_err() { break; }
            }
            from_device = incoming.next() => {
                let Some(Ok(message)) = from_device else { break };
                match message {
                    Message::Binary(payload) => {
                        if sender.send(payload).is_err() { break; }
                    }
                    Message::Ping(payload) => {
                        if outgoing.send(Message::Pong(payload)).await.is_err() { break; }
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) | Message::Text(_) => break,
                }
            }
        }
    }
}

fn keepalive_timer(interval: Option<Duration>) -> Option<tokio::time::Interval> {
    interval.map(|interval| {
        let mut timer = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        timer
    })
}

async fn next_keepalive(timer: &mut Option<tokio::time::Interval>) {
    match timer {
        Some(timer) => {
            timer.tick().await;
        }
        None => std::future::pending().await,
    }
}

enum HostFrame {
    Ciphertext {
        connection_id: ConnectionId,
        payload: Vec<u8>,
    },
    Disconnect(ConnectionId),
}

fn decode_host_frame(bytes: Vec<u8>) -> Option<HostFrame> {
    if bytes.len() < 16 {
        return None;
    }
    let id = bytes[..16].try_into().ok()?;
    let connection_id = ConnectionId::from_bytes(id);
    if bytes.len() == 16 {
        Some(HostFrame::Disconnect(connection_id))
    } else {
        Some(HostFrame::Ciphertext {
            connection_id,
            payload: bytes[16..].to_vec(),
        })
    }
}

#[derive(Deserialize)]
struct RegisterPushRequest {
    route_id: String,
    platform_token: String,
}

#[derive(Serialize)]
struct RegisterPushResponse {
    notification_capability: String,
}

async fn register_push(
    State(state): State<RelayHttpState>,
    headers: HeaderMap,
    Json(request): Json<RegisterPushRequest>,
) -> Result<(StatusCode, Json<RegisterPushResponse>), StatusCode> {
    let route = RouteId::parse(&request.route_id).map_err(map_http_error)?;
    let presented_route = bearer(&headers)
        .ok_or(StatusCode::NOT_FOUND)
        .and_then(|value| RouteId::parse(value).map_err(map_http_error))?;
    if presented_route != route {
        return Err(StatusCode::NOT_FOUND);
    }
    state
        .router
        .authorize_device(route)
        .map_err(map_http_error)?;
    let notification = state
        .router
        .register_notification(&request.platform_token)
        .map_err(map_http_error)?;
    Ok((
        StatusCode::CREATED,
        Json(RegisterPushResponse {
            notification_capability: notification.expose_once(),
        }),
    ))
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NotifyCategory {
    ApprovalRequired,
    TurnFinished,
    HostAttention,
}

impl From<NotifyCategory> for PushCategory {
    fn from(category: NotifyCategory) -> Self {
        match category {
            NotifyCategory::ApprovalRequired => Self::ApprovalRequired,
            NotifyCategory::TurnFinished => Self::TurnFinished,
            NotifyCategory::HostAttention => Self::HostAttention,
        }
    }
}

#[derive(Deserialize)]
struct NotifyPushRequest {
    category: NotifyCategory,
}

async fn notify_push(
    State(state): State<RelayHttpState>,
    headers: HeaderMap,
    Json(request): Json<NotifyPushRequest>,
) -> Result<StatusCode, StatusCode> {
    let capability = bearer(&headers)
        .ok_or(StatusCode::NOT_FOUND)
        .and_then(|value| NotificationCapability::parse(value).map_err(map_http_error))?;
    state
        .router
        .notify(capability, request.category.into())
        .map_err(map_http_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn revoke_push(
    State(state): State<RelayHttpState>,
    Path(capability): Path<String>,
) -> Result<StatusCode, StatusCode> {
    state
        .router
        .revoke_notification(NotificationCapability::parse(&capability).map_err(map_http_error)?)
        .map_err(map_http_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .filter(|value| !value.is_empty())
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or_default()
                ^ right.get(index).copied().unwrap_or_default(),
        );
    }
    difference == 0
}

fn map_http_error(error: RelayError) -> StatusCode {
    match error {
        RelayError::HostAlreadyConnected | RelayError::HostOffline => StatusCode::CONFLICT,
        RelayError::DeviceLimit | RelayError::Backpressure => StatusCode::TOO_MANY_REQUESTS,
        RelayError::InvalidFrame => StatusCode::PAYLOAD_TOO_LARGE,
        RelayError::PushUnavailable | RelayError::PersistenceUnavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        RelayError::RouteUnavailable | RelayError::Unauthorized | RelayError::Closed => {
            StatusCode::NOT_FOUND
        }
    }
}

pub fn empty_response(status: StatusCode) -> Response {
    Response::builder()
        .status(status)
        .body(Body::empty())
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::body::to_bytes;
    use axum::http::Request;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::http::Request as WsRequest;
    use tokio_tungstenite::tungstenite::Message as WsMessage;
    use tower::ServiceExt;

    use super::*;
    use crate::{PushProvider, UnavailablePush};

    const ENROLLMENT_TOKEN: &str = "development-enrollment-token-32-bytes-minimum";

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

    fn state() -> RelayHttpState {
        RelayHttpState::new(
            RelayRouter::new(Arc::new(UnavailablePush)),
            ENROLLMENT_TOKEN.into(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn enrollment_requires_the_exact_bearer_and_returns_separate_capabilities() {
        let state = state();
        let unauthorized = app(state.clone())
            .oneshot(
                Request::post("/v1/routes/enroll")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let response = app(state)
            .oneshot(
                Request::post("/v1/routes/enroll")
                    .header("authorization", format!("Bearer {ENROLLMENT_TOKEN}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert!(value.get("device_capability").is_none());
        let route_id = value["route_id"].as_str().unwrap();
        assert_eq!(route_id.len(), 22);
        assert_eq!(
            RouteId::parse(route_id).unwrap().expose_for_enrollment(),
            route_id
        );
    }

    #[tokio::test]
    async fn loopback_websockets_route_binary_records_by_connection_id() {
        let state = state();
        let enrolled = state.router.enroll_route().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });

        let host_request = ws_request(
            format!(
                "ws://{address}/v1/routes/{}/host",
                enrolled.route_id.expose_for_enrollment()
            ),
            &enrolled.host_capability.expose_for_enrollment(),
        );
        let (mut host_socket, _) = connect_async(host_request).await.unwrap();
        let device_request = device_ws_request(format!(
            "ws://{address}/v1/connect/{}",
            enrolled.route_id.expose_for_enrollment()
        ));
        let (mut device_socket, _) = connect_async(device_request).await.unwrap();

        let opaque = vec![9_u8; 128];
        device_socket
            .send(WsMessage::Binary(opaque.clone()))
            .await
            .unwrap();
        let WsMessage::Binary(at_host) = host_socket.next().await.unwrap().unwrap() else {
            panic!("host expected a binary frame")
        };
        assert_eq!(&at_host[16..], opaque.as_slice());
        let connection_id: [u8; 16] = at_host[..16].try_into().unwrap();

        let mut reply = connection_id.to_vec();
        reply.extend_from_slice(&opaque);
        host_socket.send(WsMessage::Binary(reply)).await.unwrap();
        let WsMessage::Binary(at_device) = device_socket.next().await.unwrap().unwrap() else {
            panic!("device expected a binary frame")
        };
        assert_eq!(at_device, opaque);
        server.abort();
    }

    #[tokio::test]
    async fn configured_keepalive_pings_host_and_device_sockets() {
        let state = state().with_keepalive_interval(Duration::from_millis(10));
        let enrolled = state.router.enroll_route().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app(state)).await.unwrap();
        });

        let (mut host_socket, _) = connect_async(ws_request(
            format!(
                "ws://{address}/v1/routes/{}/host",
                enrolled.route_id.expose_for_enrollment()
            ),
            &enrolled.host_capability.expose_for_enrollment(),
        ))
        .await
        .unwrap();
        let (mut device_socket, _) = connect_async(device_ws_request(format!(
            "ws://{address}/v1/connect/{}",
            enrolled.route_id.expose_for_enrollment()
        )))
        .await
        .unwrap();

        let host_message = tokio::time::timeout(Duration::from_secs(1), host_socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let device_message = tokio::time::timeout(Duration::from_secs(1), device_socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(host_message, WsMessage::Ping(_)));
        assert!(matches!(device_message, WsMessage::Ping(_)));
        server.abort();
    }

    fn ws_request(url: String, capability: &str) -> WsRequest<()> {
        WsRequest::builder()
            .uri(url)
            .header("host", "127.0.0.1")
            .header("authorization", format!("Bearer {capability}"))
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .body(())
            .unwrap()
    }

    fn device_ws_request(url: String) -> WsRequest<()> {
        WsRequest::builder()
            .uri(url)
            .header("host", "127.0.0.1")
            .header("connection", "Upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
            .body(())
            .unwrap()
    }

    #[tokio::test]
    async fn push_registration_exposes_no_caller_supplied_notification_body() {
        let push = Arc::new(FakePush::default());
        let router = RelayRouter::new(push.clone());
        let route = router.enroll_route().unwrap();
        router.authorize_device(route.route_id).unwrap();
        let capability = router.register_notification("platform-token").unwrap();
        router
            .notify(capability, PushCategory::TurnFinished)
            .unwrap();
        assert_eq!(
            *push.0.lock().unwrap(),
            [("platform-token".into(), PushCategory::TurnFinished)]
        );
    }

    #[tokio::test]
    async fn push_registration_requires_the_exact_qr_route_token() {
        let state = state();
        let route = state.router.enroll_route().unwrap();
        let route_id = route.route_id.expose_for_enrollment();
        let body = serde_json::to_vec(&serde_json::json!({
            "route_id": route_id,
            "platform_token": "platform-token"
        }))
        .unwrap();
        let wrong_route = RelayRouter::new(Arc::new(UnavailablePush))
            .enroll_route()
            .unwrap()
            .route_id
            .expose_for_enrollment();
        let denied = app(state.clone())
            .oneshot(
                Request::post("/v1/push/register")
                    .header("authorization", format!("Bearer {wrong_route}"))
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::NOT_FOUND);

        let accepted = app(state)
            .oneshot(
                Request::post("/v1/push/register")
                    .header("authorization", format!("Bearer {route_id}"))
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::CREATED);
    }
}
