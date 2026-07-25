//! Volatile one-time pairing authority for the local remote host.

#![cfg_attr(not(test), allow(dead_code))]

use std::sync::{Arc, Mutex};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use camelid_remote_protocol::{PairRequest, PairingQr, ProtocolError};
use camelid_remote_store::{RemoteStore, StoreError, StoredHostIdentity};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const PAIRING_LIFETIME_MS: u64 = 5 * 60 * 1000;
const MAX_SECRET_FAILURES: u8 = 5;

#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    #[error("pairing is unavailable")]
    Unavailable,
    #[error("pairing request is invalid")]
    Invalid,
    #[error("pairing offer has expired")]
    Expired,
    #[error("pairing request is unauthorized")]
    Unauthorized,
    #[error("pairing confirmation does not match the pending device")]
    ConfirmationMismatch,
    #[error("pairing was rejected locally")]
    Rejected,
}

impl From<ProtocolError> for PairingError {
    fn from(_: ProtocolError) -> Self {
        Self::Invalid
    }
}

impl From<StoreError> for PairingError {
    fn from(_: StoreError) -> Self {
        Self::Unavailable
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingPairing {
    pub confirmation_id: Uuid,
    pub connection_id: Uuid,
    pub device_label: String,
    pub authentication_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingStatus {
    Offered {
        expires_at_unix_ms: u64,
    },
    AwaitingConfirmation {
        confirmation_id: Uuid,
        connection_id: Uuid,
        expires_at_unix_ms: u64,
        device_label: String,
        authentication_fingerprint: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairedDevice {
    pub device_id: Uuid,
}

enum PairingState {
    Offered {
        secret: [u8; 16],
        expires_at_unix_ms: u64,
        failed_attempts: u8,
    },
    AwaitingConfirmation {
        confirmation_id: Uuid,
        connection_id: Uuid,
        expires_at_unix_ms: u64,
        device_label: String,
        device_noise_public: [u8; 32],
        authentication_fingerprint: String,
    },
}

pub struct PairingCoordinator {
    store: Arc<Mutex<RemoteStore>>,
    host: StoredHostIdentity,
    active: Mutex<Option<PairingState>>,
}

impl PairingCoordinator {
    pub fn new(store: Arc<Mutex<RemoteStore>>, host: StoredHostIdentity) -> Self {
        Self {
            store,
            host,
            active: Mutex::new(None),
        }
    }

    pub fn host_id(&self) -> Uuid {
        self.host.host_id
    }

    pub fn create_offer(
        &self,
        relay_url: &str,
        route_id: &str,
        now_unix_ms: u64,
    ) -> Result<PairingQr, PairingError> {
        let mut active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        if active.is_some() {
            return Err(PairingError::Unavailable);
        }
        let expires_at_unix_ms = now_unix_ms
            .checked_add(PAIRING_LIFETIME_MS)
            .ok_or(PairingError::Invalid)?;
        let mut secret = [0_u8; 16];
        getrandom::fill(&mut secret).map_err(|_| PairingError::Unavailable)?;
        let qr = PairingQr {
            v: 1,
            relay_url: relay_url.into(),
            route_id: route_id.into(),
            host_id: self.host.host_id,
            host_noise_public: URL_SAFE_NO_PAD.encode(self.host.noise_public),
            pairing_secret: URL_SAFE_NO_PAD.encode(secret),
            expires_at_unix_ms,
        };
        qr.validate()?;
        *active = Some(PairingState::Offered {
            secret,
            expires_at_unix_ms,
            failed_attempts: 0,
        });
        Ok(qr)
    }

    pub fn status(&self) -> Result<Option<PairingStatus>, PairingError> {
        let active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        Ok(match active.as_ref() {
            Some(PairingState::Offered {
                expires_at_unix_ms, ..
            }) => Some(PairingStatus::Offered {
                expires_at_unix_ms: *expires_at_unix_ms,
            }),
            Some(PairingState::AwaitingConfirmation {
                confirmation_id,
                connection_id,
                expires_at_unix_ms,
                device_label,
                authentication_fingerprint,
                ..
            }) => Some(PairingStatus::AwaitingConfirmation {
                confirmation_id: *confirmation_id,
                connection_id: *connection_id,
                expires_at_unix_ms: *expires_at_unix_ms,
                device_label: device_label.clone(),
                authentication_fingerprint: authentication_fingerprint.clone(),
            }),
            None => None,
        })
    }

    pub fn expire(&self, now_unix_ms: u64) -> Result<Option<Uuid>, PairingError> {
        let mut active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        let expired = match active.as_ref() {
            Some(PairingState::Offered {
                expires_at_unix_ms, ..
            }) => now_unix_ms >= *expires_at_unix_ms,
            Some(PairingState::AwaitingConfirmation {
                expires_at_unix_ms, ..
            }) => now_unix_ms >= *expires_at_unix_ms,
            None => false,
        };
        if !expired {
            return Ok(None);
        }
        let connection_id = match active.take() {
            Some(PairingState::AwaitingConfirmation { connection_id, .. }) => Some(connection_id),
            _ => None,
        };
        Ok(connection_id)
    }

    pub fn cancel(&self) -> Result<Option<Uuid>, PairingError> {
        let active = self
            .active
            .lock()
            .map_err(|_| PairingError::Unavailable)?
            .take();
        Ok(match active {
            Some(PairingState::AwaitingConfirmation { connection_id, .. }) => Some(connection_id),
            _ => None,
        })
    }

    pub fn cancel_connection(&self, connection_id: Uuid) -> Result<bool, PairingError> {
        let mut active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        let matches = matches!(
            active.as_ref(),
            Some(PairingState::AwaitingConfirmation {
                connection_id: expected,
                ..
            }) if *expected == connection_id
        );
        if matches {
            active.take();
        }
        Ok(matches)
    }

    pub fn receive_authenticated_request(
        &self,
        encoded_request: &[u8],
        connection_id: Uuid,
        device_noise_public: [u8; 32],
        handshake_hash: &[u8],
        now_unix_ms: u64,
    ) -> Result<PendingPairing, PairingError> {
        let request = PairRequest::decode(encoded_request)?;
        if handshake_hash.len() < 32 {
            return Err(PairingError::Invalid);
        }
        let presented = URL_SAFE_NO_PAD
            .decode(&request.pairing_secret)
            .map_err(|_| PairingError::Unauthorized)?;
        let mut active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        let Some(PairingState::Offered {
            secret,
            expires_at_unix_ms,
            mut failed_attempts,
        }) = active.take()
        else {
            return Err(PairingError::Unavailable);
        };
        if now_unix_ms >= expires_at_unix_ms {
            return Err(PairingError::Expired);
        }
        if !constant_time_equal(&presented, &secret) {
            failed_attempts += 1;
            if failed_attempts < MAX_SECRET_FAILURES {
                *active = Some(PairingState::Offered {
                    secret,
                    expires_at_unix_ms,
                    failed_attempts,
                });
            }
            return Err(PairingError::Unauthorized);
        }

        let confirmation_id = Uuid::new_v4();
        let authentication_fingerprint = fingerprint(handshake_hash);
        let pending = PendingPairing {
            confirmation_id,
            connection_id,
            device_label: request.device_label.clone(),
            authentication_fingerprint: authentication_fingerprint.clone(),
        };
        *active = Some(PairingState::AwaitingConfirmation {
            confirmation_id,
            connection_id,
            expires_at_unix_ms,
            device_label: request.device_label,
            device_noise_public,
            authentication_fingerprint,
        });
        Ok(pending)
    }

    pub fn confirm(
        &self,
        confirmation_id: Uuid,
        connection_id: Uuid,
        authentication_fingerprint: &str,
        accepted: bool,
        now_unix_ms: u64,
    ) -> Result<PairedDevice, PairingError> {
        let mut active = self.active.lock().map_err(|_| PairingError::Unavailable)?;
        let Some(PairingState::AwaitingConfirmation {
            confirmation_id: expected_id,
            connection_id: expected_connection_id,
            expires_at_unix_ms,
            device_label,
            device_noise_public,
            authentication_fingerprint: expected_fingerprint,
        }) = active.take()
        else {
            return Err(PairingError::Unavailable);
        };
        if confirmation_id != expected_id
            || connection_id != expected_connection_id
            || authentication_fingerprint != expected_fingerprint
        {
            return Err(PairingError::ConfirmationMismatch);
        }
        if now_unix_ms >= expires_at_unix_ms {
            return Err(PairingError::Expired);
        }
        if !accepted {
            return Err(PairingError::Rejected);
        }
        let device_id = Uuid::new_v4();
        self.store
            .lock()
            .map_err(|_| PairingError::Unavailable)?
            .register_device(device_id, &device_label, &device_noise_public, now_unix_ms)?;
        Ok(PairedDevice { device_id })
    }
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(left_byte ^ right_byte);
    }
    difference == 0
}

fn fingerprint(handshake_hash: &[u8]) -> String {
    let digest = Sha256::digest(handshake_hash);
    digest[..8]
        .chunks_exact(2)
        .map(|chunk| format!("{:02X}{:02X}", chunk[0], chunk[1]))
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn setup() -> (PairingCoordinator, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let mut store = RemoteStore::open(&directory.path().join("pairing.sqlite3")).unwrap();
        let host = StoredHostIdentity {
            host_id: Uuid::new_v4(),
            noise_public: [3_u8; 32],
            secret_reference: "os-secret://camelid/host".into(),
        };
        store
            .initialize_host_identity(host.host_id, &host.noise_public, &host.secret_reference, 1)
            .unwrap();
        (
            PairingCoordinator::new(Arc::new(Mutex::new(store)), host),
            directory,
        )
    }

    fn request(secret: &str, label: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "pairing_secret": secret,
            "device_label": label,
            "app_protocol_version": 1,
            "supported_capabilities": ["agent_events"]
        }))
        .unwrap()
    }

    #[test]
    fn pairing_requires_local_confirmation_before_device_authority() {
        let (coordinator, _directory) = setup();
        let qr = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        let device_key = [9_u8; 32];
        let connection_id = Uuid::new_v4();
        let pending = coordinator
            .receive_authenticated_request(
                &request(&qr.pairing_secret, "Phone"),
                connection_id,
                device_key,
                &[5_u8; 32],
                11,
            )
            .unwrap();
        assert!(!coordinator
            .store
            .lock()
            .unwrap()
            .device_authorized(Uuid::nil(), &device_key)
            .unwrap());
        let paired = coordinator
            .confirm(
                pending.confirmation_id,
                connection_id,
                &pending.authentication_fingerprint,
                true,
                12,
            )
            .unwrap();
        assert!(coordinator
            .store
            .lock()
            .unwrap()
            .device_authorized(paired.device_id, &device_key)
            .unwrap());
        assert!(matches!(
            coordinator.receive_authenticated_request(
                &request(&qr.pairing_secret, "Second"),
                Uuid::new_v4(),
                [8_u8; 32],
                &[6_u8; 32],
                13,
            ),
            Err(PairingError::Unavailable)
        ));
    }

    #[test]
    fn expiry_rejection_cancel_and_bad_attempts_destroy_pairing_authority() {
        let (coordinator, _directory) = setup();
        let qr = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        assert!(matches!(
            coordinator.receive_authenticated_request(
                &request(&qr.pairing_secret, "Phone"),
                Uuid::new_v4(),
                [9_u8; 32],
                &[5_u8; 32],
                qr.expires_at_unix_ms,
            ),
            Err(PairingError::Expired)
        ));

        let qr = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                20,
            )
            .unwrap();
        let wrong_secret = URL_SAFE_NO_PAD.encode([4_u8; 16]);
        for _ in 0..MAX_SECRET_FAILURES {
            assert!(matches!(
                coordinator.receive_authenticated_request(
                    &request(&wrong_secret, "Phone"),
                    Uuid::new_v4(),
                    [9_u8; 32],
                    &[5_u8; 32],
                    21,
                ),
                Err(PairingError::Unauthorized)
            ));
        }
        assert!(matches!(
            coordinator.receive_authenticated_request(
                &request(&qr.pairing_secret, "Phone"),
                Uuid::new_v4(),
                [9_u8; 32],
                &[5_u8; 32],
                22,
            ),
            Err(PairingError::Unavailable)
        ));

        coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                30,
            )
            .unwrap();
        coordinator.cancel().unwrap();
        assert!(matches!(
            coordinator.confirm(Uuid::new_v4(), Uuid::new_v4(), "0000", true, 31),
            Err(PairingError::Unavailable)
        ));
    }

    #[test]
    fn active_offer_requires_expiry_or_explicit_cancel_before_replacement() {
        let (coordinator, _directory) = setup();
        let first = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        assert!(matches!(
            coordinator.create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                11,
            ),
            Err(PairingError::Unavailable)
        ));
        assert_eq!(
            coordinator.status().unwrap(),
            Some(PairingStatus::Offered {
                expires_at_unix_ms: first.expires_at_unix_ms,
            })
        );
        assert_eq!(coordinator.expire(first.expires_at_unix_ms).unwrap(), None);
        assert!(coordinator.status().unwrap().is_none());
        assert!(coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                first.expires_at_unix_ms + 1,
            )
            .is_ok());
    }

    #[test]
    fn pending_confirmation_is_bound_to_connection_and_cancel_reports_it() {
        let (coordinator, _directory) = setup();
        let qr = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        let connection_id = Uuid::new_v4();
        let pending = coordinator
            .receive_authenticated_request(
                &request(&qr.pairing_secret, "Phone"),
                connection_id,
                [9_u8; 32],
                &[5_u8; 32],
                11,
            )
            .unwrap();
        assert_eq!(
            coordinator.status().unwrap(),
            Some(PairingStatus::AwaitingConfirmation {
                confirmation_id: pending.confirmation_id,
                connection_id,
                expires_at_unix_ms: qr.expires_at_unix_ms,
                device_label: "Phone".into(),
                authentication_fingerprint: pending.authentication_fingerprint,
            })
        );
        assert!(!coordinator.cancel_connection(Uuid::new_v4()).unwrap());
        assert!(coordinator.cancel_connection(connection_id).unwrap());
        assert!(coordinator.status().unwrap().is_none());
    }

    #[test]
    fn local_confirmation_is_bound_to_connection_and_noise_transcript() {
        let (coordinator, _directory) = setup();
        let qr = coordinator
            .create_offer(
                "wss://relay.example.invalid/v1/connect",
                "AAAAAAAAAAAAAAAAAAAAAA",
                10,
            )
            .unwrap();
        let connection_id = Uuid::new_v4();
        let pending = coordinator
            .receive_authenticated_request(
                &request(&qr.pairing_secret, "Phone"),
                connection_id,
                [9_u8; 32],
                &[5_u8; 32],
                11,
            )
            .unwrap();
        assert!(matches!(
            coordinator.confirm(
                pending.confirmation_id,
                Uuid::new_v4(),
                &pending.authentication_fingerprint,
                true,
                12,
            ),
            Err(PairingError::ConfirmationMismatch)
        ));
        assert!(matches!(
            coordinator.confirm(
                pending.confirmation_id,
                connection_id,
                &pending.authentication_fingerprint,
                true,
                13,
            ),
            Err(PairingError::Unavailable)
        ));
    }
}
