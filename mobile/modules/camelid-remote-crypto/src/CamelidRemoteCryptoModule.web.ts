import { NativeModule, registerWebModule } from 'expo';

class CamelidRemoteCryptoModule extends NativeModule {
	async bindingStatusAsync() {
		return {
			available: false,
			code: 'native_core_not_linked' as const,
			suite: 'Noise_IK_25519_ChaChaPoly_BLAKE2s' as const,
		};
	}
}

export default registerWebModule(CamelidRemoteCryptoModule, 'CamelidRemoteCrypto');
