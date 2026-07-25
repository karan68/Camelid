import { useFocusEffect, useRouter } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import CamelidRemoteCrypto from '../../modules/camelid-remote-crypto';
import { fonts, palette } from '@/constants/remote-theme';
import type { StoredHost } from '@/remote/hostStore';
import { createSecureHostStore } from '@/remote/secureHostStore';

type SecurityState = 'checking' | 'ready' | 'blocked';

export default function HostsScreen() {
  const router = useRouter();
  const [hosts, setHosts] = useState<readonly StoredHost[]>([]);
  const [loading, setLoading] = useState(true);
  const [security, setSecurity] = useState<SecurityState>('checking');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void Promise.all([createSecureHostStore(), CamelidRemoteCrypto.bindingStatusAsync()])
      .then(async ([store, status]) => {
        const storedHosts = await store.list();
        if (!active) return;
        setHosts(storedHosts);
        setSecurity(status.available ? 'ready' : 'blocked');
      })
      .catch(() => {
        if (!active) return;
        setSecurity('blocked');
        setError('Protected native services are unavailable.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useFocusEffect(useCallback(() => {
    let active = true;
    void createSecureHostStore()
      .then((store) => store.list())
      .then((storedHosts) => {
        if (active) setHosts(storedHosts);
      })
      .catch(() => {
        if (active) setError('Protected host metadata is unavailable.');
      });
    return () => {
      active = false;
    };
  }, []));

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Text style={styles.brand}>Camelid</Text>
          <View style={styles.statusRow} accessibilityLabel={`Security status: ${security}`}>
            <View style={[styles.statusDot, security === 'ready' && styles.statusDotReady]} />
            <Text style={styles.statusText}>
              {security === 'ready'
                ? 'Protected transport ready'
                : security === 'checking'
                  ? 'Checking native security'
                  : 'Development core unavailable'}
            </Text>
          </View>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Hosts</Text>
          <Text style={styles.count}>{hosts.length}</Text>
        </View>

        {loading ? (
          <View style={styles.centerState}>
            <ActivityIndicator color={palette.accent} />
          </View>
        ) : hosts.length === 0 ? (
          <View style={styles.emptyState}>
            <View style={styles.emptyIcon}>
              <HostIcon size={28} color={palette.secondary} />
            </View>
            <Text style={styles.emptyTitle}>No paired hosts</Text>
            <Text style={styles.emptyBody}>Your local agent sessions will appear here.</Text>
          </View>
        ) : (
          <View style={styles.hostList}>
            {hosts.map((host) => (
              <Pressable
                key={host.hostId}
                onPress={() => router.push({ pathname: '/host/[hostId]', params: { hostId: host.hostId } })}
                style={({ pressed }) => [styles.hostRow, pressed && styles.pressed]}>
                <View style={styles.hostGlyph}>
                  <HostIcon size={20} color={palette.accent} />
                </View>
                <View style={styles.hostText}>
                  <Text numberOfLines={1} style={styles.hostName}>
                    {host.label}
                  </Text>
                  <Text numberOfLines={1} style={styles.hostMeta}>
                    Offline · sequence {host.lastAppliedSequence}
                  </Text>
                </View>
                <SymbolView
                  name={{ ios: 'chevron.right', android: 'chevron_right', web: 'chevron_right' }}
                  size={16}
                  tintColor={palette.secondary}
                />
              </Pressable>
            ))}
          </View>
        )}

        {error !== null && <Text style={styles.error}>{error}</Text>}
      </ScrollView>

      <View style={styles.actionBar}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Pair a host"
          onPress={() => router.push('/pair')}
          style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryPressed]}>
          <SymbolView
            name={{ ios: 'qrcode.viewfinder', android: 'qr_code_scanner', web: 'qr_code_scanner' }}
            size={20}
            tintColor="#FFFFFF"
          />
          <Text style={styles.primaryLabel}>Pair host</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

function HostIcon({ size, color }: { size: number; color: string }) {
  return (
    <SymbolView
      name={{ ios: 'desktopcomputer', android: 'computer', web: 'computer' }}
      size={size}
      tintColor={color}
    />
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: palette.canvas },
  scroll: { flex: 1 },
  content: { flexGrow: 1, paddingHorizontal: 20, paddingTop: 20, paddingBottom: 24 },
  header: { gap: 8, paddingBottom: 34 },
  brand: { color: palette.ink, fontFamily: fonts.display, fontSize: 38, lineHeight: 42 },
  statusRow: { alignItems: 'center', flexDirection: 'row', gap: 8 },
  statusDot: { backgroundColor: palette.warning, borderRadius: 4, height: 8, width: 8 },
  statusDotReady: { backgroundColor: palette.accent },
  statusText: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 13 },
  sectionHeader: { alignItems: 'center', flexDirection: 'row', gap: 8, paddingBottom: 10 },
  sectionTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 15 },
  count: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 13 },
  centerState: { alignItems: 'center', justifyContent: 'center', minHeight: 180 },
  emptyState: {
    alignItems: 'center',
    borderTopColor: palette.line,
    borderTopWidth: 1,
    paddingHorizontal: 24,
    paddingTop: 52,
  },
  emptyIcon: {
    alignItems: 'center',
    backgroundColor: palette.quiet,
    borderRadius: 8,
    height: 52,
    justifyContent: 'center',
    marginBottom: 18,
    width: 52,
  },
  emptyTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 17 },
  emptyBody: {
    color: palette.secondary,
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 20,
    marginTop: 5,
    textAlign: 'center',
  },
  hostList: { borderTopColor: palette.line, borderTopWidth: 1 },
  hostRow: {
    alignItems: 'center',
    borderBottomColor: palette.line,
    borderBottomWidth: 1,
    flexDirection: 'row',
    gap: 12,
    minHeight: 70,
  },
  hostGlyph: {
    alignItems: 'center',
    backgroundColor: palette.quiet,
    borderRadius: 6,
    height: 38,
    justifyContent: 'center',
    width: 38,
  },
  hostText: { flex: 1, minWidth: 0 },
  hostName: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 15 },
  hostMeta: { color: palette.secondary, fontFamily: fonts.body, fontSize: 12, marginTop: 3 },
  pressed: { opacity: 0.65 },
  error: {
    color: palette.danger,
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
    marginTop: 18,
  },
  actionBar: {
    backgroundColor: palette.canvas,
    borderTopColor: palette.line,
    borderTopWidth: 1,
    paddingBottom: 24,
    paddingHorizontal: 20,
    paddingTop: 14,
  },
  primaryButton: {
    alignItems: 'center',
    backgroundColor: palette.accent,
    borderRadius: 6,
    flexDirection: 'row',
    gap: 9,
    justifyContent: 'center',
    minHeight: 50,
  },
  primaryPressed: { backgroundColor: palette.accentPressed },
  primaryLabel: { color: '#FFFFFF', fontFamily: fonts.semibold, fontSize: 15 },
});
