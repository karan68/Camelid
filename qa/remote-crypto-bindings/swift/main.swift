import Foundation

struct Connection {
    let initiator: TransportSession
    let responder: TransportSession
    let firstMessage: Data
    let handshakeHash: Data
}

private func bytes(_ text: String) -> Data {
    Data(text.utf8)
}

private func keypair() throws -> (publicKey: Data, privateKey: Data) {
    let generated = try generateStaticKey()
    let publicKey = try generated.publicKey()
    let privateKey = try generated.takePrivateKey()
    precondition(publicKey.count == 32 && privateKey.count == 32)
    expectCryptoError { _ = try generated.takePrivateKey() }
    expectCryptoError { _ = try generated.publicKey() }
    try generated.invalidate()
    return (publicKey, privateKey)
}

private func connect(
    hostPublic: Data,
    hostPrivate: Data,
    devicePublic: Data,
    devicePrivate: Data
) throws -> Connection {
    let initiator = try HandshakeSession.initiator(
        localPrivate: devicePrivate,
        pinnedHostPublic: hostPublic
    )
    let responder = try HandshakeSession.responder(hostPrivate: hostPrivate)
    let first = try initiator.write(payload: bytes("pair.request"))
    let requestPayload = try responder.read(message: first)
    precondition(requestPayload == bytes("pair.request"))
    let responderRemote = try responder.remoteStatic()
    precondition(responderRemote == devicePublic)
    let reply = try responder.write(payload: bytes("pair.accepted"))
    let responsePayload = try initiator.read(message: reply)
    precondition(responsePayload == bytes("pair.accepted"))
    let initiatorRemote = try initiator.remoteStatic()
    precondition(initiatorRemote == hostPublic)
    let initiatorFinished = try initiator.isFinished()
    let responderFinished = try responder.isFinished()
    precondition(initiatorFinished && responderFinished)
    let hash = try initiator.handshakeHash()
    let responderHash = try responder.handshakeHash()
    precondition(hash == responderHash)
    return Connection(
        initiator: try initiator.intoTransport(),
        responder: try responder.intoTransport(),
        firstMessage: first,
        handshakeHash: hash
    )
}

private func expectCryptoError(_ operation: () throws -> Void) {
    do {
        try operation()
        preconditionFailure("expected a cryptographic error")
    } catch is CryptoBindingError {
    } catch {
        preconditionFailure("unexpected error type")
    }
}

var host = try keypair()
var device = try keypair()

let connected = try connect(
    hostPublic: host.publicKey,
    hostPrivate: host.privateKey,
    devicePublic: device.publicKey,
    devicePrivate: device.privateKey
)
let command = try Data(contentsOf: URL(fileURLWithPath: "tests/fixtures/remote/v1/valid/start_turn_message.json"))
let sealedCommand = try connected.initiator.seal(plaintext: command)
let openedCommand = try connected.responder.open(ciphertext: sealedCommand)
precondition(openedCommand == command)
let event = bytes("canonical event")
let sealedEvent = try connected.responder.seal(plaintext: event)
let openedEvent = try connected.initiator.open(ciphertext: sealedEvent)
precondition(openedEvent == event)

let tamperedConnection = try connect(
    hostPublic: host.publicKey,
    hostPrivate: host.privateKey,
    devicePublic: device.publicKey,
    devicePrivate: device.privateKey
)
var tampered = try tamperedConnection.initiator.seal(plaintext: bytes("tamper"))
tampered[tampered.index(before: tampered.endIndex)] ^= 1
expectCryptoError { _ = try tamperedConnection.responder.open(ciphertext: tampered) }
try tamperedConnection.initiator.invalidate()
try tamperedConnection.responder.invalidate()

let wrongHost = try keypair()
let wrongInitiator = try HandshakeSession.initiator(
    localPrivate: device.privateKey,
    pinnedHostPublic: wrongHost.publicKey
)
let correctResponder = try HandshakeSession.responder(hostPrivate: host.privateKey)
expectCryptoError {
    _ = try correctResponder.read(message: wrongInitiator.write(payload: bytes("pair.request")))
}
try wrongInitiator.invalidate()
try correctResponder.invalidate()

let rekeyed = try connect(
    hostPublic: host.publicKey,
    hostPrivate: host.privateKey,
    devicePublic: device.publicKey,
    devicePrivate: device.privateKey
)
try rekeyed.initiator.rekeyOutgoing()
try rekeyed.responder.rekeyIncoming()
let afterRekey = bytes("after rekey")
let sealedAfterRekey = try rekeyed.initiator.seal(plaintext: afterRekey)
let openedAfterRekey = try rekeyed.responder.open(ciphertext: sealedAfterRekey)
precondition(openedAfterRekey == afterRekey)

let reconnect = try connect(
    hostPublic: host.publicKey,
    hostPrivate: host.privateKey,
    devicePublic: device.publicKey,
    devicePrivate: device.privateKey
)
precondition(reconnect.firstMessage != connected.firstMessage)
precondition(reconnect.handshakeHash != connected.handshakeHash)

for transport in [
    connected.initiator,
    connected.responder,
    rekeyed.initiator,
    rekeyed.responder,
    reconnect.initiator,
    reconnect.responder,
] {
    try transport.invalidate()
}
expectCryptoError { _ = try connected.initiator.seal(plaintext: Data()) }

host.privateKey.resetBytes(in: 0..<host.privateKey.count)
device.privateKey.resetBytes(in: 0..<device.privateKey.count)
print("SWIFT_REMOTE_CRYPTO_INTEROP=PASS")