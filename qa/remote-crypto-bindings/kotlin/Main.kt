import java.io.File
import ai.camelid.remote.crypto.CryptoBindingException
import ai.camelid.remote.crypto.HandshakeSession
import ai.camelid.remote.crypto.TransportSession
import ai.camelid.remote.crypto.generateStaticKey

private data class Connection(
    val initiator: TransportSession,
    val responder: TransportSession,
    val firstMessage: ByteArray,
    val handshakeHash: ByteArray,
)

private fun keypair(): Pair<ByteArray, ByteArray> {
    val generated = generateStaticKey()
    val publicKey = generated.publicKey()
    val privateKey = generated.takePrivateKey()
    check(publicKey.size == 32 && privateKey.size == 32)
    expectCryptoError { generated.takePrivateKey() }
    expectCryptoError { generated.publicKey() }
    generated.invalidate()
    return publicKey to privateKey
}

private fun connect(
    hostPublic: ByteArray,
    hostPrivate: ByteArray,
    devicePublic: ByteArray,
    devicePrivate: ByteArray,
): Connection {
    val initiator = HandshakeSession.initiator(devicePrivate, hostPublic)
    val responder = HandshakeSession.responder(hostPrivate)
    val first = initiator.write("pair.request".encodeToByteArray())
    check(responder.read(first).contentEquals("pair.request".encodeToByteArray()))
    check(responder.remoteStatic().contentEquals(devicePublic))
    val reply = responder.write("pair.accepted".encodeToByteArray())
    check(initiator.read(reply).contentEquals("pair.accepted".encodeToByteArray()))
    check(initiator.remoteStatic().contentEquals(hostPublic))
    check(initiator.isFinished() && responder.isFinished())
    val hash = initiator.handshakeHash()
    check(hash.contentEquals(responder.handshakeHash()))
    return Connection(initiator.intoTransport(), responder.intoTransport(), first, hash)
}

private fun expectCryptoError(block: () -> Unit) {
    try {
        block()
        error("expected a cryptographic error")
    } catch (_: CryptoBindingException) {
    }
}

fun main() {
    val (hostPublic, hostPrivate) = keypair()
    val (devicePublic, devicePrivate) = keypair()

    val connected = connect(hostPublic, hostPrivate, devicePublic, devicePrivate)
    val command = File("tests/fixtures/remote/v1/valid/start_turn_message.json").readBytes()
    check(connected.responder.open(connected.initiator.seal(command)).contentEquals(command))
    val event = "canonical event".encodeToByteArray()
    check(connected.initiator.open(connected.responder.seal(event)).contentEquals(event))

    val tamperedConnection = connect(hostPublic, hostPrivate, devicePublic, devicePrivate)
    val tampered = tamperedConnection.initiator.seal("tamper".encodeToByteArray())
    tampered[tampered.lastIndex] = (tampered.last().toInt() xor 1).toByte()
    expectCryptoError { tamperedConnection.responder.open(tampered) }
    tamperedConnection.initiator.invalidate()
    tamperedConnection.responder.invalidate()

    val (wrongHostPublic, _) = keypair()
    val wrongInitiator = HandshakeSession.initiator(devicePrivate, wrongHostPublic)
    val correctResponder = HandshakeSession.responder(hostPrivate)
    expectCryptoError { correctResponder.read(wrongInitiator.write("pair.request".encodeToByteArray())) }
    wrongInitiator.invalidate()
    correctResponder.invalidate()

    val rekeyed = connect(hostPublic, hostPrivate, devicePublic, devicePrivate)
    rekeyed.initiator.rekeyOutgoing()
    rekeyed.responder.rekeyIncoming()
    val afterRekey = "after rekey".encodeToByteArray()
    check(rekeyed.responder.open(rekeyed.initiator.seal(afterRekey)).contentEquals(afterRekey))

    val reconnect = connect(hostPublic, hostPrivate, devicePublic, devicePrivate)
    check(!reconnect.firstMessage.contentEquals(connected.firstMessage))
    check(!reconnect.handshakeHash.contentEquals(connected.handshakeHash))

    listOf(
        connected.initiator,
        connected.responder,
        rekeyed.initiator,
        rekeyed.responder,
        reconnect.initiator,
        reconnect.responder,
    ).forEach { it.invalidate() }
    expectCryptoError { connected.initiator.seal(ByteArray(0)) }

    hostPrivate.fill(0)
    devicePrivate.fill(0)
    println("KOTLIN_REMOTE_CRYPTO_INTEROP=PASS")
}