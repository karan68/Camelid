package ai.camelid.remote.crypto.expo

import android.util.Base64
import ai.camelid.remote.crypto.CryptoBindingException
import ai.camelid.remote.crypto.HandshakeSession
import ai.camelid.remote.crypto.TransportSession
import ai.camelid.remote.crypto.uniffiEnsureInitialized
import expo.modules.kotlin.exception.CodedException
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

private class CryptoCoreUnavailableException(cause: Throwable? = null) : CodedException(
  "The verified Camelid Rust crypto core is unavailable.",
  cause
)

private class CryptoInvalidHandleException : CodedException("The cryptographic session is closed or invalid.")
private class CryptoInvalidKeyException : CodedException("The cryptographic key is invalid or unavailable.")
private class CryptoMessageTooLargeException : CodedException("The cryptographic message exceeds the protocol bound.")
private class CryptoAuthenticationFailedException : CodedException("Remote cryptographic authentication failed.")

class CamelidRemoteCryptoModule : Module() {
  private val handshakes = ConcurrentHashMap<String, HandshakeSession>()
  private val transports = ConcurrentHashMap<String, TransportSession>()

  override fun definition() = ModuleDefinition {
    Name("CamelidRemoteCrypto")

    OnDestroy {
      closeAll()
    }

    AsyncFunction("bindingStatusAsync") {
      try {
        uniffiEnsureInitialized()
        mapOf(
          "available" to true,
          "code" to "ready",
          "suite" to "Noise_IK_25519_ChaChaPoly_BLAKE2s"
        )
      } catch (_: Throwable) {
        mapOf(
          "available" to false,
          "code" to "native_core_not_linked",
          "suite" to "Noise_IK_25519_ChaChaPoly_BLAKE2s"
        )
      }
    }

    AsyncFunction("createDeviceIdentityAsync") { hostId: String ->
      crypto {
        val identity = vault().create(hostId)
        mapOf(
          "keyReference" to identity.keyReference,
          "publicKey" to identity.publicKey,
          "protection" to "android_keystore_wrapped"
        )
      }
    }

    AsyncFunction("removeDeviceIdentityAsync") { keyReference: String ->
      crypto { vault().remove(keyReference) }
    }

    AsyncFunction("startInitiatorAsync") { keyReference: String, pinnedHostPublic: String ->
      crypto {
        val privateKey = vault().load(keyReference)
        try {
          val handshake = HandshakeSession.initiator(privateKey, decode(pinnedHostPublic))
          newHandle(handshakes, handshake)
        } finally {
          privateKey.fill(0)
        }
      }
    }

    AsyncFunction("handshakeWriteAsync") { handle: String, payload: String ->
      crypto { encode(handshake(handle).write(payload.toByteArray(Charsets.UTF_8))) }
    }

    AsyncFunction("handshakeReadAsync") { handle: String, record: String ->
      crypto { decodeUtf8(handshake(handle).read(decode(record))) }
    }

    AsyncFunction("handshakeHashAsync") { handle: String ->
      crypto { encode(handshake(handle).handshakeHash()) }
    }

    AsyncFunction("finishHandshakeAsync") { handle: String ->
      crypto {
        val session = handshakes.remove(handle) ?: throw CryptoInvalidHandleException()
        try {
          newHandle(transports, session.intoTransport())
        } finally {
          session.destroy()
        }
      }
    }

    AsyncFunction("sealAsync") { handle: String, plaintext: String ->
      crypto { encode(transport(handle).seal(decode(plaintext))) }
    }

    AsyncFunction("openAsync") { handle: String, ciphertext: String ->
      crypto { encode(transport(handle).open(decode(ciphertext))) }
    }

    AsyncFunction("rekeyOutgoingAsync") { handle: String ->
      crypto { transport(handle).rekeyOutgoing() }
    }

    AsyncFunction("rekeyIncomingAsync") { handle: String ->
      crypto { transport(handle).rekeyIncoming() }
    }

    AsyncFunction("invalidateAsync") { handle: String ->
      crypto {
        handshakes.remove(handle)?.let {
          it.invalidate()
          it.destroy()
        }
        transports.remove(handle)?.let {
          it.invalidate()
          it.destroy()
        }
      }
    }
  }

  private fun vault(): DeviceKeyVault {
    val context = appContext.reactContext?.applicationContext
      ?: throw CryptoCoreUnavailableException()
    return DeviceKeyVault(context)
  }

  private fun handshake(handle: String): HandshakeSession =
    handshakes[requireHandle(handle)] ?: throw CryptoInvalidHandleException()

  private fun transport(handle: String): TransportSession =
    transports[requireHandle(handle)] ?: throw CryptoInvalidHandleException()

  private fun requireHandle(handle: String): String {
    try {
      UUID.fromString(handle)
      return handle
    } catch (_: IllegalArgumentException) {
      throw CryptoInvalidHandleException()
    }
  }

  private fun <T> newHandle(registry: ConcurrentHashMap<String, T>, value: T): String {
    val handle = UUID.randomUUID().toString()
    registry[handle] = value
    return handle
  }

  private fun closeAll() {
    handshakes.values.forEach {
      runCatching { it.invalidate() }
      it.destroy()
    }
    transports.values.forEach {
      runCatching { it.invalidate() }
      it.destroy()
    }
    handshakes.clear()
    transports.clear()
  }

  private inline fun <T> crypto(block: () -> T): T {
    try {
      uniffiEnsureInitialized()
      return block()
    } catch (error: CodedException) {
      throw error
    } catch (_: CryptoBindingException.AuthenticationFailed) {
      throw CryptoAuthenticationFailedException()
    } catch (_: CryptoBindingException.MessageTooLarge) {
      throw CryptoMessageTooLargeException()
    } catch (_: CryptoBindingException.InvalidKey) {
      throw CryptoInvalidKeyException()
    } catch (_: CryptoBindingException.InvalidState) {
      throw CryptoInvalidHandleException()
    } catch (error: Throwable) {
      throw CryptoCoreUnavailableException(error)
    }
  }

  private fun encode(bytes: ByteArray): String = Base64.encodeToString(
    bytes,
    Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING
  )

  private fun decode(value: String): ByteArray = try {
    Base64.decode(value, Base64.URL_SAFE or Base64.NO_WRAP)
  } catch (_: IllegalArgumentException) {
    throw CryptoInvalidKeyException()
  }

  private fun decodeUtf8(bytes: ByteArray): String = try {
    Charsets.UTF_8.newDecoder()
      .onMalformedInput(CodingErrorAction.REPORT)
      .onUnmappableCharacter(CodingErrorAction.REPORT)
      .decode(ByteBuffer.wrap(bytes))
      .toString()
  } catch (_: Exception) {
    throw CryptoAuthenticationFailedException()
  } finally {
    bytes.fill(0)
  }
}
