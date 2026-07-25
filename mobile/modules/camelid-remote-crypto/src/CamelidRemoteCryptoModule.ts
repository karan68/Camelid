import { NativeModule, requireNativeModule } from 'expo';

import type {
  Base64UrlData,
  CryptoBindingStatus,
  DeviceIdentity,
  HandshakeHandle,
  KeyReference,
  TransportHandle,
} from './CamelidRemoteCrypto.types';

declare class CamelidRemoteCryptoModule extends NativeModule {
  bindingStatusAsync(): Promise<CryptoBindingStatus>;
  createDeviceIdentityAsync(hostId: string): Promise<DeviceIdentity>;
  removeDeviceIdentityAsync(keyReference: KeyReference): Promise<void>;
  startInitiatorAsync(keyReference: KeyReference, pinnedHostPublic: string): Promise<HandshakeHandle>;
  handshakeWriteAsync(handle: HandshakeHandle, payload: string): Promise<string>;
  handshakeReadAsync(handle: HandshakeHandle, record: string): Promise<string>;
  handshakeHashAsync(handle: HandshakeHandle): Promise<string>;
  finishHandshakeAsync(handle: HandshakeHandle): Promise<TransportHandle>;
  sealAsync(handle: TransportHandle, plaintext: Base64UrlData): Promise<Base64UrlData>;
  openAsync(handle: TransportHandle, ciphertext: Base64UrlData): Promise<Base64UrlData>;
  rekeyOutgoingAsync(handle: TransportHandle): Promise<void>;
  rekeyIncomingAsync(handle: TransportHandle): Promise<void>;
  invalidateAsync(handle: HandshakeHandle | TransportHandle): Promise<void>;
}

export default requireNativeModule<CamelidRemoteCryptoModule>('CamelidRemoteCrypto');
