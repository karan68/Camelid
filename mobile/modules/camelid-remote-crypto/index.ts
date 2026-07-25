// Re-export the native module. On web, it will be resolved to CamelidRemoteCryptoModule.web.ts
// and on native platforms to CamelidRemoteCryptoModule.ts
export { default } from './src/CamelidRemoteCryptoModule';
export * from './src/CamelidRemoteCrypto.types';
