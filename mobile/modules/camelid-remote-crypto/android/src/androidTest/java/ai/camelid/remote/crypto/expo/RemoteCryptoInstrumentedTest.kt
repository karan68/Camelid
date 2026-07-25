package ai.camelid.remote.crypto.expo

import ai.camelid.remote.crypto.HandshakeSession
import ai.camelid.remote.crypto.generateStaticKey
import ai.camelid.remote.crypto.uniffiEnsureInitialized
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.util.UUID
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RemoteCryptoInstrumentedTest {
  @Test
  fun keystoreWrappedIdentityCompletesNoiseRoundTripAndRevocation() {
    uniffiEnsureInitialized()
    val vault = DeviceKeyVault(ApplicationProvider.getApplicationContext())
    val hostId = UUID.randomUUID().toString()
    val identity = vault.create(hostId)
    val host = generateStaticKey()
    val privateKey = vault.load(identity.keyReference)
    val hostPublic = host.publicKey()
    val hostPrivate = host.takePrivateKey()
    try {
      val initiator = HandshakeSession.initiator(privateKey, hostPublic)
      val responder = HandshakeSession.responder(hostPrivate)
      try {
        val request = initiator.write("pair.request".toByteArray())
        assertFalse(request.contentEquals("pair.request".toByteArray()))
        assertArrayEquals("pair.request".toByteArray(), responder.read(request))
        val response = responder.write("pair.accepted".toByteArray())
        assertArrayEquals("pair.accepted".toByteArray(), initiator.read(response))
        val deviceTransport = initiator.intoTransport()
        val hostTransport = responder.intoTransport()
        try {
          val ciphertext = deviceTransport.seal("command".toByteArray())
          assertArrayEquals("command".toByteArray(), hostTransport.open(ciphertext))
        } finally {
          deviceTransport.invalidate()
          hostTransport.invalidate()
          deviceTransport.destroy()
          hostTransport.destroy()
        }
      } finally {
        initiator.destroy()
        responder.destroy()
      }
    } finally {
      privateKey.fill(0)
      hostPrivate.fill(0)
      host.invalidate()
      host.destroy()
      vault.remove(identity.keyReference)
    }
    assertTrue(runCatching { vault.load(identity.keyReference) }.isFailure)
  }
}
