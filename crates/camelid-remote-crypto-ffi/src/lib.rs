//! Opaque Swift/Kotlin adapter for the fixed-suite remote cryptography core.
//!
//! Foreign callers never select Noise primitives and never receive raw `snow`
//! state. Errors carry no key, frame, or plaintext detail. Authentication
//! failures are connection-terminal: callers must close and reconnect.

use std::sync::{Arc, Mutex, MutexGuard};

use camelid_remote_crypto::{RemoteCryptoError, RemoteHandshake, RemoteTransport, StaticKeypair};

uniffi::setup_scaffolding!();

#[derive(Debug, thiserror::Error, uniffi::Error, PartialEq, Eq)]
pub enum CryptoBindingError {
    #[error("remote cryptography is unavailable")]
    Unavailable,
    #[error("invalid remote cryptographic key")]
    InvalidKey,
    #[error("invalid remote cryptographic state")]
    InvalidState,
    #[error("remote cryptographic message is too large")]
    MessageTooLarge,
    #[error("remote cryptographic authentication failed")]
    AuthenticationFailed,
}

impl From<RemoteCryptoError> for CryptoBindingError {
    fn from(error: RemoteCryptoError) -> Self {
        match error {
            RemoteCryptoError::Unavailable => Self::Unavailable,
            RemoteCryptoError::InvalidKey => Self::InvalidKey,
            RemoteCryptoError::InvalidState => Self::InvalidState,
            RemoteCryptoError::MessageTooLarge => Self::MessageTooLarge,
            RemoteCryptoError::AuthenticationFailed => Self::AuthenticationFailed,
        }
    }
}

#[derive(uniffi::Object)]
pub struct GeneratedStaticKey {
    inner: Mutex<Option<StaticKeypair>>,
}

#[uniffi::export]
pub fn generate_static_key() -> Result<Arc<GeneratedStaticKey>, CryptoBindingError> {
    Ok(Arc::new(GeneratedStaticKey {
        inner: Mutex::new(Some(StaticKeypair::generate()?)),
    }))
}

#[uniffi::export]
impl GeneratedStaticKey {
    pub fn public_key(&self) -> Result<Vec<u8>, CryptoBindingError> {
        lock(&self.inner)?
            .as_ref()
            .map(|keypair| keypair.public().to_vec())
            .ok_or(CryptoBindingError::InvalidState)
    }

    /// Consume the private key exactly once. The caller must move the returned
    /// bytes immediately into Keychain/Keystore-backed storage and clear its
    /// transient buffer. This object cannot expose either key afterward.
    pub fn take_private_key(&self) -> Result<Vec<u8>, CryptoBindingError> {
        lock(&self.inner)?
            .take()
            .map(|keypair| keypair.private().to_vec())
            .ok_or(CryptoBindingError::InvalidState)
    }

    pub fn invalidate(&self) -> Result<(), CryptoBindingError> {
        lock(&self.inner)?.take();
        Ok(())
    }
}

#[derive(uniffi::Object)]
pub struct HandshakeSession {
    inner: Mutex<Option<RemoteHandshake>>,
}

#[uniffi::export]
impl HandshakeSession {
    #[uniffi::constructor]
    pub fn initiator(
        local_private: Vec<u8>,
        pinned_host_public: Vec<u8>,
    ) -> Result<Arc<Self>, CryptoBindingError> {
        let local_private = key32(local_private)?;
        let pinned_host_public = key32(pinned_host_public)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(Some(RemoteHandshake::initiator(
                &local_private,
                &pinned_host_public,
            )?)),
        }))
    }

    #[uniffi::constructor]
    pub fn responder(host_private: Vec<u8>) -> Result<Arc<Self>, CryptoBindingError> {
        let host_private = key32(host_private)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(Some(RemoteHandshake::responder(&host_private)?)),
        }))
    }

    pub fn write(&self, payload: Vec<u8>) -> Result<Vec<u8>, CryptoBindingError> {
        handshake(&mut lock(&self.inner)?)?
            .write(&payload)
            .map_err(Into::into)
    }

    pub fn read(&self, message: Vec<u8>) -> Result<Vec<u8>, CryptoBindingError> {
        handshake(&mut lock(&self.inner)?)?
            .read(&message)
            .map_err(Into::into)
    }

    pub fn is_finished(&self) -> Result<bool, CryptoBindingError> {
        Ok(handshake(&mut lock(&self.inner)?)?.is_finished())
    }

    pub fn handshake_hash(&self) -> Result<Vec<u8>, CryptoBindingError> {
        Ok(handshake(&mut lock(&self.inner)?)?.handshake_hash())
    }

    pub fn remote_static(&self) -> Result<Vec<u8>, CryptoBindingError> {
        handshake(&mut lock(&self.inner)?)?
            .remote_static()
            .map(|key| key.to_vec())
            .ok_or(CryptoBindingError::InvalidState)
    }

    pub fn into_transport(&self) -> Result<Arc<TransportSession>, CryptoBindingError> {
        let inner = lock(&self.inner)?
            .take()
            .ok_or(CryptoBindingError::InvalidState)?
            .into_transport()?;
        Ok(Arc::new(TransportSession {
            inner: Mutex::new(Some(inner)),
        }))
    }

    pub fn invalidate(&self) -> Result<(), CryptoBindingError> {
        lock(&self.inner)?.take();
        Ok(())
    }
}

#[derive(uniffi::Object)]
pub struct TransportSession {
    inner: Mutex<Option<RemoteTransport>>,
}

#[uniffi::export]
impl TransportSession {
    pub fn seal(&self, plaintext: Vec<u8>) -> Result<Vec<u8>, CryptoBindingError> {
        transport(&mut lock(&self.inner)?)?
            .seal(&plaintext)
            .map_err(Into::into)
    }

    pub fn open(&self, ciphertext: Vec<u8>) -> Result<Vec<u8>, CryptoBindingError> {
        transport(&mut lock(&self.inner)?)?
            .open(&ciphertext)
            .map_err(Into::into)
    }

    pub fn rekey_outgoing(&self) -> Result<(), CryptoBindingError> {
        transport(&mut lock(&self.inner)?)?.rekey_outgoing();
        Ok(())
    }

    pub fn rekey_incoming(&self) -> Result<(), CryptoBindingError> {
        transport(&mut lock(&self.inner)?)?.rekey_incoming();
        Ok(())
    }

    pub fn remote_static(&self) -> Result<Vec<u8>, CryptoBindingError> {
        transport(&mut lock(&self.inner)?)?
            .remote_static()
            .map(|key| key.to_vec())
            .ok_or(CryptoBindingError::InvalidState)
    }

    pub fn invalidate(&self) -> Result<(), CryptoBindingError> {
        lock(&self.inner)?.take();
        Ok(())
    }
}

fn key32(key: Vec<u8>) -> Result<[u8; 32], CryptoBindingError> {
    key.try_into().map_err(|_| CryptoBindingError::InvalidKey)
}

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, CryptoBindingError> {
    mutex.lock().map_err(|_| CryptoBindingError::InvalidState)
}

fn handshake<'state>(
    state: &'state mut MutexGuard<'_, Option<RemoteHandshake>>,
) -> Result<&'state mut RemoteHandshake, CryptoBindingError> {
    state.as_mut().ok_or(CryptoBindingError::InvalidState)
}

fn transport<'state>(
    state: &'state mut MutexGuard<'_, Option<RemoteTransport>>,
) -> Result<&'state mut RemoteTransport, CryptoBindingError> {
    state.as_mut().ok_or(CryptoBindingError::InvalidState)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connected() -> (Arc<TransportSession>, Arc<TransportSession>) {
        let host = generate_static_key().unwrap();
        let host_public = host.public_key().unwrap();
        let host_private = host.take_private_key().unwrap();
        let device = generate_static_key().unwrap();
        let device_public = device.public_key().unwrap();
        let device_private = device.take_private_key().unwrap();
        let initiator = HandshakeSession::initiator(device_private, host_public.clone()).unwrap();
        let responder = HandshakeSession::responder(host_private).unwrap();
        let first = initiator.write(b"pair.request".to_vec()).unwrap();
        assert_eq!(responder.read(first).unwrap(), b"pair.request");
        let reply = responder.write(b"pair.accepted".to_vec()).unwrap();
        assert_eq!(initiator.read(reply).unwrap(), b"pair.accepted");
        assert_eq!(initiator.remote_static().unwrap(), host_public);
        assert_eq!(responder.remote_static().unwrap(), device_public);
        assert_eq!(
            initiator.handshake_hash().unwrap(),
            responder.handshake_hash().unwrap()
        );
        (
            initiator.into_transport().unwrap(),
            responder.into_transport().unwrap(),
        )
    }

    #[test]
    fn opaque_adapter_completes_handshake_transport_and_rekey() {
        let (initiator, responder) = connected();
        let ciphertext = initiator.seal(b"command".to_vec()).unwrap();
        assert_eq!(responder.open(ciphertext).unwrap(), b"command");
        initiator.rekey_outgoing().unwrap();
        responder.rekey_incoming().unwrap();
        let ciphertext = initiator.seal(b"after rekey".to_vec()).unwrap();
        assert_eq!(responder.open(ciphertext).unwrap(), b"after rekey");
    }

    #[test]
    fn close_is_idempotent_and_blocks_future_use() {
        let (initiator, _) = connected();
        initiator.invalidate().unwrap();
        initiator.invalidate().unwrap();
        assert_eq!(
            initiator.seal(Vec::new()),
            Err(CryptoBindingError::InvalidState)
        );
    }

    #[test]
    fn adapter_errors_are_bounded_and_secret_free() {
        assert!(matches!(
            HandshakeSession::initiator(vec![7; 31], vec![8; 32]),
            Err(CryptoBindingError::InvalidKey)
        ));
        for error in [
            CryptoBindingError::Unavailable,
            CryptoBindingError::InvalidKey,
            CryptoBindingError::InvalidState,
            CryptoBindingError::MessageTooLarge,
            CryptoBindingError::AuthenticationFailed,
        ] {
            let message = error.to_string();
            assert!(message.len() < 80);
            assert!(!message.contains('7'));
            assert!(!message.contains('8'));
        }
    }

    #[test]
    fn generated_private_key_is_one_shot_and_closes_the_public_view() {
        let key = generate_static_key().unwrap();
        assert_eq!(key.public_key().unwrap().len(), 32);
        assert_eq!(key.take_private_key().unwrap().len(), 32);
        assert_eq!(
            key.take_private_key(),
            Err(CryptoBindingError::InvalidState)
        );
        assert_eq!(key.public_key(), Err(CryptoBindingError::InvalidState));
        key.invalidate().unwrap();
    }
}
