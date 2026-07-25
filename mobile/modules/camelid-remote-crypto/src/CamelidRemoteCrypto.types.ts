export interface CryptoBindingStatus {
	available: boolean;
	code: 'ready' | 'native_core_not_linked';
	suite: 'Noise_IK_25519_ChaChaPoly_BLAKE2s';
}

declare const keyReferenceBrand: unique symbol;
declare const handshakeHandleBrand: unique symbol;
declare const transportHandleBrand: unique symbol;
declare const binaryDataBrand: unique symbol;

export type KeyReference = string & { readonly [keyReferenceBrand]: true };
export type HandshakeHandle = string & { readonly [handshakeHandleBrand]: true };
export type TransportHandle = string & { readonly [transportHandleBrand]: true };
export type Base64UrlData = string & { readonly [binaryDataBrand]: true };

export interface DeviceIdentity {
	keyReference: KeyReference;
	publicKey: string;
	protection: 'ios_keychain_device_only' | 'android_keystore_wrapped';
}
