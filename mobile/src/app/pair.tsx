import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import { useEffect, useState } from 'react';
import {
  KeyboardAvoidingView,
  ActivityIndicator,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import CamelidRemoteCrypto from '../../modules/camelid-remote-crypto';
import { fonts, palette } from '@/constants/remote-theme';
import { pairHost } from '@/remote/pairingClient';
import { PairingQr, PairingQrError, pairingSocketUrl, parsePairingQr } from '@/remote/pairingQr';
import { createSecureHostStore } from '@/remote/secureHostStore';

export default function PairHostScreen() {
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const [qr, setQr] = useState<PairingQr | null>(null);
  const [label, setLabel] = useState('My phone');
  const [error, setError] = useState<string | null>(null);
  const [cryptoReady, setCryptoReady] = useState(false);
  const [pairing, setPairing] = useState(false);

  useEffect(() => {
    let active = true;
    void CamelidRemoteCrypto.bindingStatusAsync().then((status) => {
      if (active) setCryptoReady(status.available);
    });
    return () => {
      active = false;
    };
  }, []);

  const onScanned = ({ data }: { data: string }) => {
    if (qr !== null) return;
    try {
      setQr(parsePairingQr(data, Date.now()));
      setError(null);
    } catch (caught) {
      setError(caught instanceof PairingQrError ? caught.message : 'Pairing QR is invalid.');
    }
  };

  const reset = () => {
    if (pairing) return;
    setQr(null);
    setError(null);
  };

  const continuePairing = async () => {
    if (qr === null || pairing || !cryptoReady) return;
    setPairing(true);
    setError(null);
    try {
      const hostStore = await createSecureHostStore();
      await pairHost({
        qr,
        deviceLabel: label,
        crypto: CamelidRemoteCrypto,
        hostStore,
      });
      setQr(null);
      router.dismiss();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Pairing failed.');
    } finally {
      setPairing(false);
    }
  };

  return (
    <SafeAreaView edges={['bottom']} style={styles.safeArea}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          {qr === null ? (
            <View style={styles.scannerSection}>
              {permission?.granted ? (
                <View style={styles.cameraFrame}>
                  <CameraView
                    barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
                    facing="back"
                    onBarcodeScanned={onScanned}
                    style={StyleSheet.absoluteFill}
                  />
                  <View pointerEvents="none" style={styles.scanTarget} />
                </View>
              ) : (
                <View style={styles.permissionState}>
                  <SymbolView
                    name={{ ios: 'camera', android: 'photo_camera', web: 'photo_camera' }}
                    size={30}
                    tintColor={palette.secondary}
                  />
                  <Text style={styles.permissionTitle}>Camera access required</Text>
                  <Pressable
                    accessibilityRole="button"
                    onPress={() => void requestPermission()}
                    style={({ pressed }) => [styles.secondaryButton, pressed && styles.pressed]}>
                    <Text style={styles.secondaryLabel}>Allow camera</Text>
                  </Pressable>
                </View>
              )}
              {error !== null && <Text style={styles.error}>{error}</Text>}
            </View>
          ) : (
            <View style={styles.preview}>
              <View style={styles.verifiedRow}>
                <View style={styles.verifiedIcon}>
                  <SymbolView
                    name={{ ios: 'checkmark.shield', android: 'verified_user', web: 'verified_user' }}
                    size={22}
                    tintColor={palette.accent}
                  />
                </View>
                <View style={styles.flex}>
                  <Text style={styles.previewTitle}>Host identity pinned</Text>
                  <Text style={styles.previewMeta}>{new URL(qr.relay_url).host}</Text>
                </View>
              </View>

              <View style={styles.details}>
                <Detail label="Host ID" value={qr.host_id} />
                <Detail label="Pinned host key" value={qr.host_noise_public} selectable />
                <Detail label="Relay route" value={pairingSocketUrl(qr)} />
              </View>

              <View style={styles.inputSection}>
                <Text style={styles.inputLabel}>Device label</Text>
                <TextInput
                  accessibilityLabel="Device label"
                  autoCapitalize="words"
                  maxLength={128}
                  onChangeText={setLabel}
                  placeholder="My phone"
                  placeholderTextColor={palette.secondary}
                  style={styles.input}
                  value={label}
                />
              </View>

              {!cryptoReady && (
                <View style={styles.blockedBand}>
                  <SymbolView
                    name={{ ios: 'lock.trianglebadge.exclamationmark', android: 'lock', web: 'lock' }}
                    size={18}
                    tintColor={palette.warning}
                  />
                  <Text style={styles.blockedText}>Verified native crypto core is not linked.</Text>
                </View>
              )}

              {error !== null && <Text style={styles.error}>{error}</Text>}

              <Pressable
                accessibilityRole="button"
                disabled={!cryptoReady || label.trim().length === 0 || pairing}
                onPress={() => void continuePairing()}
                style={({ pressed }) => [
                  styles.primaryButton,
                  (!cryptoReady || label.trim().length === 0 || pairing) && styles.disabledButton,
                  pressed && cryptoReady && !pairing && styles.primaryPressed,
                ]}>
                {pairing ? (
                  <ActivityIndicator color="#FFFFFF" />
                ) : (
                  <Text style={styles.primaryLabel}>Continue pairing</Text>
                )}
              </Pressable>
              <Pressable
                accessibilityRole="button"
                disabled={pairing}
                onPress={reset}
                style={styles.rescanButton}>
                <Text style={styles.rescanLabel}>Scan another code</Text>
              </Pressable>
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Detail({ label, value, selectable = false }: { label: string; value: string; selectable?: boolean }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text selectable={selectable} style={styles.detailValue}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  safeArea: { backgroundColor: palette.canvas, flex: 1 },
  content: { flexGrow: 1, padding: 20 },
  scannerSection: { flex: 1, gap: 16 },
  cameraFrame: {
    aspectRatio: 3 / 4,
    backgroundColor: palette.ink,
    borderRadius: 8,
    maxHeight: 560,
    overflow: 'hidden',
    width: '100%',
  },
  scanTarget: {
    alignSelf: 'center',
    borderColor: '#FFFFFF',
    borderRadius: 6,
    borderWidth: 2,
    height: '42%',
    marginTop: '48%',
    opacity: 0.9,
    width: '70%',
  },
  permissionState: {
    alignItems: 'center',
    borderColor: palette.line,
    borderRadius: 8,
    borderWidth: 1,
    gap: 16,
    justifyContent: 'center',
    minHeight: 320,
    padding: 24,
  },
  permissionTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 16 },
  secondaryButton: {
    borderColor: palette.line,
    borderRadius: 6,
    borderWidth: 1,
    minHeight: 44,
    paddingHorizontal: 18,
    justifyContent: 'center',
  },
  secondaryLabel: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 14 },
  pressed: { opacity: 0.65 },
  error: { color: palette.danger, fontFamily: fonts.medium, fontSize: 13, lineHeight: 18 },
  preview: { gap: 24 },
  verifiedRow: { alignItems: 'center', flexDirection: 'row', gap: 12 },
  verifiedIcon: {
    alignItems: 'center',
    backgroundColor: '#E5F3ED',
    borderRadius: 7,
    height: 44,
    justifyContent: 'center',
    width: 44,
  },
  previewTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 17 },
  previewMeta: { color: palette.secondary, fontFamily: fonts.body, fontSize: 13, marginTop: 2 },
  details: { borderTopColor: palette.line, borderTopWidth: 1 },
  detailRow: { borderBottomColor: palette.line, borderBottomWidth: 1, gap: 6, paddingVertical: 14 },
  detailLabel: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 12 },
  detailValue: { color: palette.ink, fontFamily: fonts.body, fontSize: 13, lineHeight: 19 },
  inputSection: { gap: 7 },
  inputLabel: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 13 },
  input: {
    backgroundColor: palette.surface,
    borderColor: palette.line,
    borderRadius: 6,
    borderWidth: 1,
    color: palette.ink,
    fontFamily: fonts.body,
    fontSize: 15,
    minHeight: 48,
    paddingHorizontal: 14,
  },
  blockedBand: {
    alignItems: 'center',
    backgroundColor: palette.warningSurface,
    borderRadius: 6,
    flexDirection: 'row',
    gap: 10,
    padding: 13,
  },
  blockedText: { color: palette.warning, flex: 1, fontFamily: fonts.medium, fontSize: 13 },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: palette.accent,
    borderRadius: 6,
    justifyContent: 'center',
    minHeight: 50,
  },
  primaryPressed: { backgroundColor: palette.accentPressed },
  disabledButton: { backgroundColor: '#AAB3AE' },
  primaryLabel: { color: '#FFFFFF', fontFamily: fonts.semibold, fontSize: 15 },
  rescanButton: { alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  rescanLabel: { color: palette.accent, fontFamily: fonts.semibold, fontSize: 14 },
});
