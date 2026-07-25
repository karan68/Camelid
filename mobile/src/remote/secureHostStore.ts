import * as SecureStore from 'expo-secure-store';

import { HostStore, HostStoreError, ProtectedValueStore } from './hostStore';

const options: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

const secureValues: ProtectedValueStore = {
  get: (key) => SecureStore.getItemAsync(key, options),
  set: (key, value) => SecureStore.setItemAsync(key, value, options),
  remove: (key) => SecureStore.deleteItemAsync(key, options),
};

export async function createSecureHostStore(): Promise<HostStore> {
  if (!(await SecureStore.isAvailableAsync())) {
    throw new HostStoreError('Protected storage is unavailable on this device.');
  }
  return new HostStore(secureValues);
}
