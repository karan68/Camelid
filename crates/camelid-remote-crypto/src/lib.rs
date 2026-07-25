//! Fixed-suite Noise core for Camelid remote agent control.
//!
//! The application cannot select a pattern or primitive. Mobile bindings expose
//! this state machine rather than `snow` directly, keeping Swift and Kotlin out
//! of handshake logic. This crate owns no relay, persistence, or agent state.

use snow::{Builder, HandshakeState, TransportState};

pub const NOISE_SUITE: &str = "Noise_IK_25519_ChaChaPoly_BLAKE2s";
pub const NOISE_PROLOGUE: &[u8] = b"camelid.remote/v1";
pub const MAX_NOISE_RECORD_BYTES: usize = 65_535;
pub const NOISE_TAG_BYTES: usize = 16;
pub const MAX_TRANSPORT_PLAINTEXT_BYTES: usize = MAX_NOISE_RECORD_BYTES - NOISE_TAG_BYTES;
pub const MAX_HANDSHAKE_PAYLOAD_BYTES: usize = 4 * 1024;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RemoteCryptoError {
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

/// A generated static keypair. Deliberately neither `Clone` nor `Debug`; the
/// secret is wiped from this owned wrapper on drop.
pub struct StaticKeypair {
    private: [u8; 32],
    public: [u8; 32],
}

impl StaticKeypair {
    pub fn generate() -> Result<Self, RemoteCryptoError> {
        let keypair = builder()?
            .generate_keypair()
            .map_err(|_| RemoteCryptoError::Unavailable)?;
        let private = keypair
            .private
            .try_into()
            .map_err(|_| RemoteCryptoError::InvalidKey)?;
        let public = keypair
            .public
            .try_into()
            .map_err(|_| RemoteCryptoError::InvalidKey)?;
        Ok(Self { private, public })
    }

    pub fn private(&self) -> &[u8; 32] {
        &self.private
    }

    pub fn public(&self) -> &[u8; 32] {
        &self.public
    }
}

impl Drop for StaticKeypair {
    fn drop(&mut self) {
        self.private.fill(0);
    }
}

pub struct RemoteHandshake {
    inner: HandshakeState,
}

impl RemoteHandshake {
    pub fn initiator(
        local_private: &[u8; 32],
        host_public: &[u8; 32],
    ) -> Result<Self, RemoteCryptoError> {
        let inner = builder()?
            .local_private_key(local_private)
            .map_err(|_| RemoteCryptoError::InvalidKey)?
            .remote_public_key(host_public)
            .map_err(|_| RemoteCryptoError::InvalidKey)?
            .prologue(NOISE_PROLOGUE)
            .map_err(|_| RemoteCryptoError::InvalidState)?
            .build_initiator()
            .map_err(|_| RemoteCryptoError::Unavailable)?;
        Ok(Self { inner })
    }

    pub fn responder(host_private: &[u8; 32]) -> Result<Self, RemoteCryptoError> {
        let inner = builder()?
            .local_private_key(host_private)
            .map_err(|_| RemoteCryptoError::InvalidKey)?
            .prologue(NOISE_PROLOGUE)
            .map_err(|_| RemoteCryptoError::InvalidState)?
            .build_responder()
            .map_err(|_| RemoteCryptoError::Unavailable)?;
        Ok(Self { inner })
    }

    pub fn write(&mut self, payload: &[u8]) -> Result<Vec<u8>, RemoteCryptoError> {
        if payload.len() > MAX_HANDSHAKE_PAYLOAD_BYTES {
            return Err(RemoteCryptoError::MessageTooLarge);
        }
        let mut message = vec![0; MAX_NOISE_RECORD_BYTES];
        let written = self
            .inner
            .write_message(payload, &mut message)
            .map_err(|_| RemoteCryptoError::InvalidState)?;
        message.truncate(written);
        Ok(message)
    }

    pub fn read(&mut self, message: &[u8]) -> Result<Vec<u8>, RemoteCryptoError> {
        if message.len() > MAX_NOISE_RECORD_BYTES {
            return Err(RemoteCryptoError::MessageTooLarge);
        }
        let mut payload = vec![0; MAX_HANDSHAKE_PAYLOAD_BYTES];
        let read = self
            .inner
            .read_message(message, &mut payload)
            .map_err(|_| RemoteCryptoError::AuthenticationFailed)?;
        payload.truncate(read);
        Ok(payload)
    }

    pub fn is_finished(&self) -> bool {
        self.inner.is_handshake_finished()
    }

    pub fn handshake_hash(&self) -> Vec<u8> {
        self.inner.get_handshake_hash().to_vec()
    }

    pub fn remote_static(&self) -> Option<[u8; 32]> {
        self.inner.get_remote_static()?.try_into().ok()
    }

    pub fn into_transport(self) -> Result<RemoteTransport, RemoteCryptoError> {
        if !self.inner.is_handshake_finished() {
            return Err(RemoteCryptoError::InvalidState);
        }
        let inner = self
            .inner
            .into_transport_mode()
            .map_err(|_| RemoteCryptoError::InvalidState)?;
        Ok(RemoteTransport { inner })
    }
}

pub struct RemoteTransport {
    inner: TransportState,
}

impl RemoteTransport {
    pub fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, RemoteCryptoError> {
        if plaintext.len() > MAX_TRANSPORT_PLAINTEXT_BYTES {
            return Err(RemoteCryptoError::MessageTooLarge);
        }
        let mut ciphertext = vec![0; MAX_NOISE_RECORD_BYTES];
        let written = self
            .inner
            .write_message(plaintext, &mut ciphertext)
            .map_err(|_| RemoteCryptoError::InvalidState)?;
        ciphertext.truncate(written);
        Ok(ciphertext)
    }

    pub fn open(&mut self, ciphertext: &[u8]) -> Result<Vec<u8>, RemoteCryptoError> {
        if ciphertext.len() > MAX_NOISE_RECORD_BYTES {
            return Err(RemoteCryptoError::MessageTooLarge);
        }
        let mut plaintext = vec![0; MAX_TRANSPORT_PLAINTEXT_BYTES];
        let read = self
            .inner
            .read_message(ciphertext, &mut plaintext)
            .map_err(|_| RemoteCryptoError::AuthenticationFailed)?;
        plaintext.truncate(read);
        Ok(plaintext)
    }

    pub fn rekey_outgoing(&mut self) {
        self.inner.rekey_outgoing();
    }

    pub fn rekey_incoming(&mut self) {
        self.inner.rekey_incoming();
    }

    pub fn remote_static(&self) -> Option<[u8; 32]> {
        self.inner.get_remote_static()?.try_into().ok()
    }
}

fn builder() -> Result<Builder<'static>, RemoteCryptoError> {
    let params = NOISE_SUITE
        .parse()
        .map_err(|_| RemoteCryptoError::Unavailable)?;
    Ok(Builder::new(params))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Connected {
        initiator: RemoteTransport,
        responder: RemoteTransport,
        first_message: Vec<u8>,
        handshake_hash: Vec<u8>,
    }

    fn connect(host: &StaticKeypair, device: &StaticKeypair) -> Connected {
        let mut initiator = RemoteHandshake::initiator(device.private(), host.public()).unwrap();
        let mut responder = RemoteHandshake::responder(host.private()).unwrap();

        let first_message = initiator.write(b"pair.request").unwrap();
        assert_eq!(responder.read(&first_message).unwrap(), b"pair.request");
        assert_eq!(responder.remote_static(), Some(*device.public()));

        let reply = responder.write(b"pair.accepted").unwrap();
        assert_eq!(initiator.read(&reply).unwrap(), b"pair.accepted");
        assert!(initiator.is_finished());
        assert!(responder.is_finished());
        assert_eq!(initiator.remote_static(), Some(*host.public()));
        assert_eq!(initiator.handshake_hash(), responder.handshake_hash());
        let handshake_hash = initiator.handshake_hash();

        Connected {
            initiator: initiator.into_transport().unwrap(),
            responder: responder.into_transport().unwrap(),
            first_message,
            handshake_hash,
        }
    }

    #[test]
    fn ik_authenticates_both_static_keys_and_exchanges_transport_records() {
        let host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();
        let mut connected = connect(&host, &device);

        let request = connected.initiator.seal(b"canonical command").unwrap();
        assert_eq!(
            connected.responder.open(&request).unwrap(),
            b"canonical command"
        );
        let event = connected.responder.seal(b"canonical event").unwrap();
        assert_eq!(
            connected.initiator.open(&event).unwrap(),
            b"canonical event"
        );
        assert_eq!(connected.initiator.remote_static(), Some(*host.public()));
        assert_eq!(connected.responder.remote_static(), Some(*device.public()));
    }

    #[test]
    fn wrong_pinned_host_key_and_tampered_handshake_fail_authentication() {
        let host = StaticKeypair::generate().unwrap();
        let wrong_host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();

        let mut initiator =
            RemoteHandshake::initiator(device.private(), wrong_host.public()).unwrap();
        let mut responder = RemoteHandshake::responder(host.private()).unwrap();
        let first = initiator.write(b"pair.request").unwrap();
        assert_eq!(
            responder.read(&first),
            Err(RemoteCryptoError::AuthenticationFailed)
        );

        let mut initiator = RemoteHandshake::initiator(device.private(), host.public()).unwrap();
        let mut responder = RemoteHandshake::responder(host.private()).unwrap();
        let mut tampered = initiator.write(b"pair.request").unwrap();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            responder.read(&tampered),
            Err(RemoteCryptoError::AuthenticationFailed)
        );
    }

    #[test]
    fn every_handshake_and_transport_byte_is_authenticated() {
        let host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();

        let mut sample_initiator =
            RemoteHandshake::initiator(device.private(), host.public()).unwrap();
        let handshake_message = sample_initiator.write(b"pair.request").unwrap();
        for offset in 0..handshake_message.len() {
            let mut initiator =
                RemoteHandshake::initiator(device.private(), host.public()).unwrap();
            let mut responder = RemoteHandshake::responder(host.private()).unwrap();
            let mut message = initiator.write(b"pair.request").unwrap();
            message[offset] ^= 1;
            assert_eq!(
                responder.read(&message),
                Err(RemoteCryptoError::AuthenticationFailed),
                "handshake byte {offset} was not authenticated"
            );
        }

        let mut sample = connect(&host, &device);
        let ciphertext = sample.initiator.seal(b"canonical command").unwrap();
        for offset in 0..ciphertext.len() {
            let mut connected = connect(&host, &device);
            let mut message = connected.initiator.seal(b"canonical command").unwrap();
            message[offset] ^= 1;
            assert_eq!(
                connected.responder.open(&message),
                Err(RemoteCryptoError::AuthenticationFailed),
                "transport byte {offset} was not authenticated"
            );
        }
    }

    #[test]
    fn transport_tamper_and_pre_rekey_ciphertext_are_rejected() {
        let host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();
        let mut tamper_connection = connect(&host, &device);

        let mut tampered = tamper_connection.initiator.seal(b"command").unwrap();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            tamper_connection.responder.open(&tampered),
            Err(RemoteCryptoError::AuthenticationFailed)
        );

        let mut stale_connection = connect(&host, &device);
        let stale = stale_connection.initiator.seal(b"before rekey").unwrap();
        stale_connection.responder.rekey_incoming();
        assert_eq!(
            stale_connection.responder.open(&stale),
            Err(RemoteCryptoError::AuthenticationFailed)
        );

        let mut rekeyed_connection = connect(&host, &device);
        rekeyed_connection.initiator.rekey_outgoing();
        rekeyed_connection.responder.rekey_incoming();
        let fresh = rekeyed_connection.initiator.seal(b"after rekey").unwrap();
        assert_eq!(
            rekeyed_connection.responder.open(&fresh).unwrap(),
            b"after rekey"
        );
    }

    #[test]
    fn reconnect_uses_fresh_ephemeral_keys_and_transcript() {
        let host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();
        let first = connect(&host, &device);
        let second = connect(&host, &device);
        assert_ne!(first.first_message, second.first_message);
        assert_ne!(first.handshake_hash, second.handshake_hash);
    }

    #[test]
    fn record_and_handshake_payload_bounds_fail_before_crypto() {
        let host = StaticKeypair::generate().unwrap();
        let device = StaticKeypair::generate().unwrap();
        let mut handshake = RemoteHandshake::initiator(device.private(), host.public()).unwrap();
        assert_eq!(
            handshake.write(&vec![0; MAX_HANDSHAKE_PAYLOAD_BYTES + 1]),
            Err(RemoteCryptoError::MessageTooLarge)
        );

        let mut connected = connect(&host, &device);
        assert_eq!(
            connected
                .initiator
                .seal(&vec![0; MAX_TRANSPORT_PLAINTEXT_BYTES + 1]),
            Err(RemoteCryptoError::MessageTooLarge)
        );
        assert_eq!(
            connected
                .responder
                .open(&vec![0; MAX_NOISE_RECORD_BYTES + 1]),
            Err(RemoteCryptoError::MessageTooLarge)
        );
    }
}
