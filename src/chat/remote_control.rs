use std::sync::{Arc, Mutex};

use camelid_remote_store::{RemoteStore, SessionState};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

#[derive(Clone)]
pub struct RemoteManagementHandle {
    store: Arc<Mutex<Option<Arc<Mutex<RemoteStore>>>>>,
    pairing: Arc<Mutex<Option<RemotePairingStatus>>>,
    commands: mpsc::UnboundedSender<RemoteManagementCommand>,
}

pub enum RemoteManagementCommand {
    CreatePairingOffer {
        reply: oneshot::Sender<Result<RemotePairingOffer, RemoteManagementError>>,
    },
    ConfirmPairing {
        confirmation_id: Uuid,
        accepted: bool,
        reply: oneshot::Sender<Result<(), RemoteManagementError>>,
    },
    CancelPairing {
        reply: oneshot::Sender<Result<(), RemoteManagementError>>,
    },
    RevokeDevice {
        device_id: Uuid,
        reply: oneshot::Sender<Result<(), RemoteManagementError>>,
    },
    EmergencyDisable {
        reply: oneshot::Sender<Result<(), RemoteManagementError>>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum RemoteManagementError {
    Unavailable,
    Rejected,
}

#[derive(Clone, serde::Serialize)]
pub struct RemoteManagementStatus {
    pub configured: bool,
    pub host_id: Option<Uuid>,
    pub relay_url: Option<String>,
    pub session: Option<RemoteManagementSession>,
    pub pairing: Option<RemotePairingStatus>,
    pub devices: Vec<RemoteManagementDevice>,
}

#[derive(Clone, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RemotePairingStatus {
    Offered {
        expires_at_unix_ms: u64,
    },
    AwaitingConfirmation {
        confirmation_id: Uuid,
        expires_at_unix_ms: u64,
        device_label: String,
        authentication_fingerprint: String,
    },
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct RemotePairingOffer {
    pub qr_payload: String,
    pub expires_at_unix_ms: u64,
}

#[derive(Clone, serde::Serialize)]
pub struct RemoteManagementSession {
    pub session_id: Uuid,
    pub state: &'static str,
    pub last_event_sequence: u64,
    pub updated_at_unix_ms: u64,
    pub capability_snapshot: serde_json::Value,
}

#[derive(Clone, serde::Serialize)]
pub struct RemoteManagementDevice {
    pub device_id: Uuid,
    pub label: String,
    pub status: &'static str,
    pub created_at_unix_ms: u64,
    pub last_seen_at_unix_ms: Option<u64>,
    pub revoked_at_unix_ms: Option<u64>,
}

pub fn channel() -> (
    RemoteManagementHandle,
    mpsc::UnboundedReceiver<RemoteManagementCommand>,
) {
    let (commands, receiver) = mpsc::unbounded_channel();
    (
        RemoteManagementHandle {
            store: Arc::new(Mutex::new(None)),
            pairing: Arc::new(Mutex::new(None)),
            commands,
        },
        receiver,
    )
}

impl RemoteManagementHandle {
    pub fn activate(&self, store: Arc<Mutex<RemoteStore>>) -> Result<(), RemoteManagementError> {
        *self
            .store
            .lock()
            .map_err(|_| RemoteManagementError::Unavailable)? = Some(store);
        Ok(())
    }

    pub fn status(&self) -> Result<RemoteManagementStatus, RemoteManagementError> {
        let pairing = self
            .pairing
            .lock()
            .map_err(|_| RemoteManagementError::Unavailable)?
            .clone();
        let store = self
            .store
            .lock()
            .map_err(|_| RemoteManagementError::Unavailable)?
            .clone();
        match store {
            Some(store) => snapshot(&store),
            None => Ok(RemoteManagementStatus {
                configured: false,
                host_id: None,
                relay_url: None,
                session: None,
                pairing: pairing.clone(),
                devices: Vec::new(),
            }),
        }
        .map(|mut status| {
            status.pairing = pairing;
            status
        })
    }

    pub fn publish_pairing(
        &self,
        pairing: Option<RemotePairingStatus>,
    ) -> Result<(), RemoteManagementError> {
        *self
            .pairing
            .lock()
            .map_err(|_| RemoteManagementError::Unavailable)? = pairing;
        Ok(())
    }

    pub async fn create_pairing_offer(&self) -> Result<RemotePairingOffer, RemoteManagementError> {
        self.request_with_result(|reply| RemoteManagementCommand::CreatePairingOffer { reply })
            .await
    }

    pub async fn confirm_pairing(
        &self,
        confirmation_id: Uuid,
        accepted: bool,
    ) -> Result<(), RemoteManagementError> {
        self.request(|reply| RemoteManagementCommand::ConfirmPairing {
            confirmation_id,
            accepted,
            reply,
        })
        .await
    }

    pub async fn cancel_pairing(&self) -> Result<(), RemoteManagementError> {
        self.request(|reply| RemoteManagementCommand::CancelPairing { reply })
            .await
    }

    pub async fn revoke_device(&self, device_id: Uuid) -> Result<(), RemoteManagementError> {
        self.request(|reply| RemoteManagementCommand::RevokeDevice { device_id, reply })
            .await
    }

    pub async fn emergency_disable(&self) -> Result<(), RemoteManagementError> {
        self.request(|reply| RemoteManagementCommand::EmergencyDisable { reply })
            .await
    }

    async fn request(
        &self,
        build: impl FnOnce(
            oneshot::Sender<Result<(), RemoteManagementError>>,
        ) -> RemoteManagementCommand,
    ) -> Result<(), RemoteManagementError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(build(reply))
            .map_err(|_| RemoteManagementError::Unavailable)?;
        response
            .await
            .map_err(|_| RemoteManagementError::Unavailable)?
    }

    async fn request_with_result<T>(
        &self,
        build: impl FnOnce(oneshot::Sender<Result<T, RemoteManagementError>>) -> RemoteManagementCommand,
    ) -> Result<T, RemoteManagementError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(build(reply))
            .map_err(|_| RemoteManagementError::Unavailable)?;
        response
            .await
            .map_err(|_| RemoteManagementError::Unavailable)?
    }
}

fn snapshot(
    store: &Arc<Mutex<RemoteStore>>,
) -> Result<RemoteManagementStatus, RemoteManagementError> {
    let store = store
        .lock()
        .map_err(|_| RemoteManagementError::Unavailable)?;
    let host_id = store
        .optional_host_identity()
        .map_err(|_| RemoteManagementError::Unavailable)?
        .map(|identity| identity.host_id);
    let relay_url = store
        .relay_binding()
        .map_err(|_| RemoteManagementError::Unavailable)?
        .map(|binding| binding.relay_url);
    let session = store
        .active_session_summary()
        .map_err(|_| RemoteManagementError::Unavailable)?
        .map(|summary| {
            Ok(RemoteManagementSession {
                session_id: summary.session_id,
                state: session_state_token(summary.state),
                last_event_sequence: summary.last_event_sequence,
                updated_at_unix_ms: summary.updated_at_unix_ms,
                capability_snapshot: serde_json::from_str(&summary.capability_snapshot_json)
                    .map_err(|_| RemoteManagementError::Unavailable)?,
            })
        })
        .transpose()?;
    let devices = store
        .devices()
        .map_err(|_| RemoteManagementError::Unavailable)?
        .into_iter()
        .map(|device| RemoteManagementDevice {
            device_id: device.device_id,
            label: device.label,
            status: if device.revoked_at_unix_ms.is_some() {
                "revoked"
            } else {
                "authorized"
            },
            created_at_unix_ms: device.created_at_unix_ms,
            last_seen_at_unix_ms: device.last_seen_at_unix_ms,
            revoked_at_unix_ms: device.revoked_at_unix_ms,
        })
        .collect();
    Ok(RemoteManagementStatus {
        configured: host_id.is_some(),
        host_id,
        relay_url,
        session,
        pairing: None,
        devices,
    })
}

fn session_state_token(state: SessionState) -> &'static str {
    match state {
        SessionState::Armed => "armed",
        SessionState::Idle => "idle",
        SessionState::Running => "running",
        SessionState::WaitingApproval => "waiting_approval",
        SessionState::Cancelling => "cancelling",
        SessionState::Failed => "failed",
        SessionState::Closed => "closed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inactive_handle_reports_unconfigured_and_rejects_actions() {
        let (handle, receiver) = channel();
        assert!(!handle.status().unwrap().configured);
        drop(receiver);
        assert!(matches!(
            handle.emergency_disable().await,
            Err(RemoteManagementError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn pairing_offer_request_returns_only_the_explicit_one_shot_response() {
        let (handle, mut receiver) = channel();
        let request = tokio::spawn({
            let handle = handle.clone();
            async move { handle.create_pairing_offer().await }
        });
        let Some(RemoteManagementCommand::CreatePairingOffer { reply }) = receiver.recv().await
        else {
            panic!("expected create pairing offer command");
        };
        reply
            .send(Ok(RemotePairingOffer {
                qr_payload: "secret-bearing-payload".into(),
                expires_at_unix_ms: 300_000,
            }))
            .unwrap();
        let offer = request.await.unwrap().unwrap();
        assert_eq!(offer.qr_payload, "secret-bearing-payload");
        assert_eq!(offer.expires_at_unix_ms, 300_000);
        assert!(handle.status().unwrap().pairing.is_none());
    }

    #[test]
    fn status_reports_durable_active_session_instead_of_latest_updated_history() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("remote.sqlite3")).unwrap();
        let active_session_id = Uuid::new_v4();
        let newer_history_id = Uuid::new_v4();
        store
            .create_session(active_session_id, "root", "model", "sha256:model", "{}", 10)
            .unwrap();
        store
            .create_session(newer_history_id, "root", "model", "sha256:model", "{}", 20)
            .unwrap();
        store.activate_session(active_session_id, 30).unwrap();

        let (handle, _receiver) = channel();
        handle.activate(Arc::new(Mutex::new(store))).unwrap();

        let session = handle.status().unwrap().session.unwrap();
        assert_eq!(session.session_id, active_session_id);
    }
}
