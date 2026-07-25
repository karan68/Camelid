package ai.camelid.remote.crypto.expo

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import ai.camelid.remote.crypto.generateStaticKey
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.security.KeyStore
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

internal data class StoredDeviceIdentity(
  val keyReference: String,
  val publicKey: String
)

internal class DeviceKeyVault(private val context: Context) {
  fun create(hostId: String): StoredDeviceIdentity {
    requireUuid(hostId)
    val reference = UUID.randomUUID().toString()
    val generated = generateStaticKey()
    var privateKey: ByteArray? = null
    try {
      val publicKey = generated.publicKey()
      privateKey = generated.takePrivateKey()
      store(reference, privateKey)
      return StoredDeviceIdentity(reference, encode(publicKey))
    } catch (error: Throwable) {
      remove(reference)
      throw error
    } finally {
      privateKey?.fill(0)
      generated.invalidate()
      generated.destroy()
    }
  }

  fun load(reference: String): ByteArray {
    requireUuid(reference)
    val file = keyFile(reference)
    if (!file.isFile || file.length() !in 34..1024) {
      throw IllegalStateException("Device key is unavailable")
    }
    val bytes = file.readBytes()
    try {
      val buffer = ByteBuffer.wrap(bytes)
      if (buffer.get() != FORMAT_VERSION) throw IllegalStateException("Device key format is invalid")
      val ivLength = buffer.int
      if (ivLength !in 12..32 || buffer.remaining() <= ivLength) {
        throw IllegalStateException("Device key format is invalid")
      }
      val iv = ByteArray(ivLength).also(buffer::get)
      val ciphertext = ByteArray(buffer.remaining()).also(buffer::get)
      val cipher = Cipher.getInstance(CIPHER)
      cipher.init(Cipher.DECRYPT_MODE, existingKey(reference), GCMParameterSpec(128, iv))
      return cipher.doFinal(ciphertext).also {
        if (it.size != PRIVATE_KEY_BYTES) {
          it.fill(0)
          throw IllegalStateException("Device key length is invalid")
        }
      }
    } finally {
      bytes.fill(0)
    }
  }

  fun remove(reference: String) {
    requireUuid(reference)
    keyFile(reference).delete()
    val keyStore = keyStore()
    if (keyStore.containsAlias(alias(reference))) keyStore.deleteEntry(alias(reference))
  }

  private fun store(reference: String, privateKey: ByteArray) {
    if (privateKey.size != PRIVATE_KEY_BYTES) throw IllegalArgumentException("Invalid private key")
    val cipher = Cipher.getInstance(CIPHER)
    cipher.init(Cipher.ENCRYPT_MODE, createKey(reference))
    val ciphertext = cipher.doFinal(privateKey)
    val bytes = ByteBuffer.allocate(1 + Int.SIZE_BYTES + cipher.iv.size + ciphertext.size)
      .put(FORMAT_VERSION)
      .putInt(cipher.iv.size)
      .put(cipher.iv)
      .put(ciphertext)
      .array()
    val destination = keyFile(reference)
    val temporary = File(destination.parentFile, "${destination.name}.tmp")
    try {
      FileOutputStream(temporary).use { stream ->
        stream.write(bytes)
        stream.fd.sync()
      }
      if (destination.exists() && !destination.delete()) throw IllegalStateException("Stale key file")
      if (!temporary.renameTo(destination)) throw IllegalStateException("Device key commit failed")
    } finally {
      bytes.fill(0)
      temporary.delete()
    }
  }

  private fun createKey(reference: String): SecretKey {
    val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
    generator.init(
      KeyGenParameterSpec.Builder(
        alias(reference),
        KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
      )
        .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
        .setKeySize(256)
        .setRandomizedEncryptionRequired(true)
        .build()
    )
    return generator.generateKey()
  }

  private fun existingKey(reference: String): SecretKey {
    return keyStore().getKey(alias(reference), null) as? SecretKey
      ?: throw IllegalStateException("Device key wrapper is unavailable")
  }

  private fun keyStore(): KeyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }

  private fun keyFile(reference: String): File {
    val directory = File(context.noBackupFilesDir, DIRECTORY)
    if (!directory.exists() && !directory.mkdirs()) throw IllegalStateException("Key directory unavailable")
    return File(directory, "$reference.bin")
  }

  private fun alias(reference: String) = "$ALIAS_PREFIX$reference"

  private fun requireUuid(value: String) {
    UUID.fromString(value)
  }

  private fun encode(bytes: ByteArray): String = Base64.encodeToString(
    bytes,
    Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING
  )

  companion object {
    private const val KEYSTORE = "AndroidKeyStore"
    private const val CIPHER = "AES/GCM/NoPadding"
    private const val DIRECTORY = "camelid-remote-device-keys"
    private const val ALIAS_PREFIX = "camelid.remote.device."
    private const val PRIVATE_KEY_BYTES = 32
    private const val FORMAT_VERSION: Byte = 1
  }
}
