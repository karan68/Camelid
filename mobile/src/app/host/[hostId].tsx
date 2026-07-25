import type { KeyReference } from '../../../modules/camelid-remote-crypto';
import CamelidRemoteCrypto from '../../../modules/camelid-remote-crypto';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { fonts, palette } from '@/constants/remote-theme';
import type { ApprovalProjection, CapabilityProjection } from '@/remote/reducer';
import { createSecureHostStore } from '@/remote/secureHostStore';
import { type RemoteSessionController, useRemoteSession } from '@/remote/sessionController';

type Tab = 'session' | 'history' | 'activity' | 'settings';

export default function HostSessionScreen() {
  const { hostId } = useLocalSearchParams<{ hostId: string }>();
  const router = useRouter();
  const remote = useRemoteSession(hostId);
  const [tab, setTab] = useState<Tab>('session');
  const [draft, setDraft] = useState('');
  const [pendingAction, setPendingAction] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [selectedApprovalId, setSelectedApprovalId] = useState<string | null>(null);
  const pendingApprovals = useMemo(
    () => remote.canControl
      ? Object.values(remote.projection.approvals).filter((approval) => approval.status === 'pending')
      : [],
    [remote.canControl, remote.projection.approvals],
  );
  const selectedApproval = selectedApprovalId === null
    ? null
    : remote.projection.approvals[selectedApprovalId] ?? null;
  const active = remote.canControl && ['running', 'waiting_approval', 'cancelling'].includes(remote.projection.status);
  const canStart = remote.canControl && remote.connection === 'connected' && !active && draft.trim().length > 0;

  const start = async () => {
    if (!canStart || pendingAction) return;
    setPendingAction(true);
    setLocalError(null);
    try {
      await remote.start(draft.trim());
      setDraft('');
    } catch (error) {
      setLocalError(message(error));
    } finally {
      setPendingAction(false);
    }
  };

  const cancel = async () => {
    if (!active || pendingAction) return;
    setPendingAction(true);
    setLocalError(null);
    try {
      await remote.cancel();
    } catch (error) {
      setLocalError(message(error));
    } finally {
      setPendingAction(false);
    }
  };

  const decide = async (approval: ApprovalProjection, decision: 'allow_once' | 'deny' | 'abort_turn') => {
    if (pendingAction || approval.status !== 'pending') return;
    setPendingAction(true);
    setLocalError(null);
    try {
      await remote.decide(approval.approvalId, decision);
      setSelectedApprovalId(null);
    } catch (error) {
      setLocalError(message(error));
    } finally {
      setPendingAction(false);
    }
  };

  const removeHost = () => {
    if (remote.host === null) return;
    Alert.alert('Remove paired host?', 'This removes the protected device key and local host record.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Remove',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            const host = remote.host;
            if (host === null) return;
            await CamelidRemoteCrypto.removeDeviceIdentityAsync(host.keyReference as KeyReference);
            const store = await createSecureHostStore();
            await store.remove(host.hostId);
            router.replace('/');
          })().catch((error) => setLocalError(message(error)));
        },
      },
    ]);
  };

  return (
    <SafeAreaView edges={['bottom']} style={styles.safeArea}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.flex}>
        <View style={styles.topBar}>
          <View style={styles.hostHeading}>
            <Text numberOfLines={1} style={styles.hostName}>{remote.host?.label ?? 'Host'}</Text>
            <ConnectionStatus connection={remote.connection} status={remote.projection.status} />
          </View>
          <Pressable
            accessibilityLabel="Reconnect"
            disabled={remote.connection === 'connecting'}
            onPress={() => void remote.connect()}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}>
            <SymbolView
              name={{ ios: 'arrow.clockwise', android: 'refresh', web: 'refresh' }}
              size={19}
              tintColor={palette.ink}
            />
          </Pressable>
        </View>

        <View style={styles.tabs}>
          {(['session', 'history', 'activity', 'settings'] as const).map((value) => (
            <Pressable
              key={value}
              accessibilityRole="tab"
              accessibilityState={{ selected: tab === value }}
              onPress={() => setTab(value)}
              style={[styles.tab, tab === value && styles.tabSelected]}>
              <Text style={[styles.tabLabel, tab === value && styles.tabLabelSelected]}>
                {capitalize(value)}
              </Text>
            </Pressable>
          ))}
        </View>

        {tab === 'session' && (
          <SessionView
            approvals={pendingApprovals}
            onApproval={setSelectedApprovalId}
            projection={remote.projection}
          />
        )}
        {tab === 'history' && (
          <HistoryView
            busy={pendingAction}
            remote={remote}
            onCreate={() => {
              setPendingAction(true);
              setLocalError(null);
              void remote.createSession()
                .then(() => setTab('session'))
                .catch((caught) => setLocalError(message(caught)))
                .finally(() => setPendingAction(false));
            }}
            onActivate={(sessionId) => {
              setPendingAction(true);
              setLocalError(null);
              void remote.activateSession(sessionId)
                .then(() => setTab('session'))
                .catch((caught) => setLocalError(message(caught)))
                .finally(() => setPendingAction(false));
            }}
            onSelect={(sessionId) => {
              void remote.selectSession(sessionId)
                .then(() => setTab('session'))
                .catch((caught) => setLocalError(message(caught)));
            }}
          />
        )}
        {tab === 'activity' && <ActivityView projection={remote.projection} />}
        {tab === 'settings' && remote.host !== null && (
          <SettingsView
            capabilities={remote.projection.capabilities}
            host={remote.host}
            onRemove={removeHost}
          />
        )}

        {(remote.error ?? localError) !== null && (
          <View style={styles.errorBand}>
            <Text style={styles.errorText}>{localError ?? remote.error}</Text>
          </View>
        )}
        {remote.commandNotice !== null && remote.error === null && localError === null && (
          <Text style={styles.commandNotice}>{remote.commandNotice}</Text>
        )}

        {tab === 'session' && remote.canControl && (
          <View style={styles.composer}>
            {active ? (
              <Pressable
                accessibilityRole="button"
                disabled={pendingAction || remote.projection.status === 'cancelling'}
                onPress={() => void cancel()}
                style={({ pressed }) => [styles.cancelButton, pressed && styles.pressed]}>
                {pendingAction ? <ActivityIndicator color={palette.danger} /> : (
                  <>
                    <SymbolView
                      name={{ ios: 'stop.fill', android: 'stop', web: 'stop' }}
                      size={17}
                      tintColor={palette.danger}
                    />
                    <Text style={styles.cancelLabel}>
                      {remote.projection.status === 'cancelling' ? 'Cancelling' : 'Cancel turn'}
                    </Text>
                  </>
                )}
              </Pressable>
            ) : (
              <View style={styles.composeRow}>
                <TextInput
                  accessibilityLabel="Agent task"
                  editable={remote.connection === 'connected'}
                  maxLength={4096}
                  multiline
                  onChangeText={setDraft}
                  placeholder="Ask the local agent"
                  placeholderTextColor={palette.secondary}
                  style={styles.composeInput}
                  value={draft}
                />
                <Pressable
                  accessibilityLabel="Start turn"
                  disabled={!canStart || pendingAction}
                  onPress={() => void start()}
                  style={({ pressed }) => [
                    styles.sendButton,
                    (!canStart || pendingAction) && styles.sendDisabled,
                    pressed && canStart && styles.primaryPressed,
                  ]}>
                  {pendingAction ? <ActivityIndicator color="#FFFFFF" /> : (
                    <SymbolView
                      name={{ ios: 'arrow.up', android: 'arrow_upward', web: 'arrow_upward' }}
                      size={20}
                      tintColor="#FFFFFF"
                    />
                  )}
                </Pressable>
              </View>
            )}
          </View>
        )}
        {tab === 'session' && !remote.canControl && (
          <View style={styles.historyFooter}>
            <Text style={styles.historyFooterText}>Replay-only history</Text>
          </View>
        )}
      </KeyboardAvoidingView>

      <ApprovalModal
        approval={selectedApproval}
        capabilities={remote.projection.capabilities}
        connected={remote.connection === 'connected'}
        connection={remote.connection}
        hostLabel={remote.host?.label ?? 'Host'}
        pendingAction={pendingAction}
        onClose={() => setSelectedApprovalId(null)}
        onDecision={decide}
      />
    </SafeAreaView>
  );
}

function HistoryView({
  busy,
  remote,
  onActivate,
  onCreate,
  onSelect,
}: {
  busy: boolean;
  remote: RemoteSessionController;
  onActivate(sessionId: string): void;
  onCreate(): void;
  onSelect(sessionId: string): void;
}) {
  return (
    <ScrollView contentContainerStyle={styles.scrollContent} style={styles.flex}>
      <View style={styles.historyHeader}>
        <Text style={styles.groupTitle}>Agent history</Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="New agent session"
          disabled={busy || !remote.canControl || remote.connection !== 'connected'}
          onPress={onCreate}
          style={({ pressed }) => [
            styles.newSessionButton,
            (busy || !remote.canControl || remote.connection !== 'connected') && styles.disabledAction,
            pressed && styles.pressed,
          ]}>
          <SymbolView
            name={{ ios: 'plus', android: 'add', web: 'add' }}
            size={16}
            tintColor={palette.accent}
          />
          <Text style={styles.newSessionLabel}>New</Text>
        </Pressable>
      </View>
      <Text style={styles.historyIntro}>
        Histories come from this host and workspace. Only the active session can run tools.
      </Text>
      {remote.catalog === null ? (
        <View style={styles.centerState}><ActivityIndicator color={palette.accent} /></View>
      ) : remote.catalog.sessions.length === 0 ? (
        <Text style={styles.muted}>No agent history yet</Text>
      ) : remote.catalog.sessions.map((session) => (
        <Pressable
          key={session.historyId}
          accessibilityRole="button"
          accessibilityLabel={`${session.title}, ${session.active ? 'active session' : 'history'}`}
          onPress={() => onSelect(session.historyId)}
          style={({ pressed }) => [
            styles.historyRow,
            session.historyId === remote.selectedSessionId && styles.historyRowSelected,
            pressed && styles.pressed,
          ]}>
          <View style={styles.flex}>
            <View style={styles.historyTitleRow}>
              <Text numberOfLines={1} style={styles.historyTitle}>{session.title}</Text>
              {session.active && <Text style={styles.activeBadge}>Active</Text>}
            </View>
            <Text numberOfLines={1} style={styles.historyMeta}>
              {session.source === 'remote' ? 'Remote agent' : 'Saved agent'} · {session.state} · event {session.lastEventSequence}
            </Text>
            {!session.active && session.continuable && (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`Continue ${session.title}`}
                disabled={busy}
                onPress={(event) => {
                  event.stopPropagation();
                  onActivate(session.historyId);
                }}
                style={({ pressed }) => [styles.continueButton, pressed && styles.pressed]}>
                <Text style={styles.continueLabel}>Continue</Text>
              </Pressable>
            )}
          </View>
          <SymbolView
            name={{ ios: 'chevron.right', android: 'chevron_right', web: 'chevron_right' }}
            size={16}
            tintColor={palette.secondary}
          />
        </Pressable>
      ))}
    </ScrollView>
  );
}

function SessionView({
  approvals,
  onApproval,
  projection,
}: {
  approvals: readonly ApprovalProjection[];
  onApproval(id: string): void;
  projection: ReturnType<typeof useRemoteSession>['projection'];
}) {
  return (
    <ScrollView contentContainerStyle={styles.scrollContent} style={styles.flex}>
      {approvals.map((approval) => (
        <Pressable
          key={approval.approvalId}
          onPress={() => onApproval(approval.approvalId)}
          style={({ pressed }) => [styles.approvalRow, pressed && styles.pressed]}>
          <View style={styles.approvalMark} />
          <View style={styles.flex}>
            <Text style={styles.approvalTitle}>Approval required</Text>
            <Text numberOfLines={1} style={styles.approvalSummary}>{approvalSummary(approval)}</Text>
          </View>
          <SymbolView
            name={{ ios: 'chevron.right', android: 'chevron_right', web: 'chevron_right' }}
            size={16}
            tintColor={palette.warning}
          />
        </Pressable>
      ))}

      {projection.transcript.length === 0 && projection.streamingAnswer.length === 0 ? (
        <View style={styles.emptyTranscript}>
          <Text style={styles.emptyTranscriptTitle}>No turns yet</Text>
        </View>
      ) : (
        projection.transcript.map((entry, index) => (
          <View key={`${entry.role}-${index}`} style={styles.message}>
            <Text style={styles.messageRole}>{entry.role === 'user' ? 'YOU' : 'CAMELID'}</Text>
            <Text selectable style={styles.messageText}>{entry.content}</Text>
          </View>
        ))
      )}
      {projection.streamingAnswer.length > 0 && (
        <View style={styles.message}>
          <Text style={styles.messageRole}>CAMELID</Text>
          <Text selectable style={styles.messageText}>{projection.streamingAnswer}</Text>
        </View>
      )}
    </ScrollView>
  );
}

function ActivityView({ projection }: { projection: ReturnType<typeof useRemoteSession>['projection'] }) {
  return (
    <ScrollView contentContainerStyle={styles.scrollContent} style={styles.flex}>
      <Text style={styles.groupTitle}>Plan</Text>
      {projection.plan.length === 0 ? <Text style={styles.muted}>No active plan</Text> : projection.plan.map((step, index) => (
        <View key={`${step.text}-${index}`} style={styles.activityRow}>
          <SymbolView
            name={step.status === 'done'
              ? { ios: 'checkmark.circle.fill', android: 'check_circle', web: 'check_circle' }
              : step.status === 'in_progress'
                ? { ios: 'circle.lefthalf.filled', android: 'pending', web: 'pending' }
                : { ios: 'circle', android: 'radio_button_unchecked', web: 'radio_button_unchecked' }}
            size={17}
            tintColor={step.status === 'done' ? palette.accent : palette.secondary}
          />
          <Text style={styles.activityText}>{step.text}</Text>
        </View>
      ))}

      <Text style={[styles.groupTitle, styles.groupSpacing]}>Tools</Text>
      {projection.tools.length === 0 ? <Text style={styles.muted}>No tool activity</Text> : [...projection.tools].reverse().map((tool) => (
        <View key={tool.callId} style={styles.toolRow}>
          <View style={styles.toolHeader}>
            <Text style={styles.toolName}>{tool.tool}</Text>
            <Text style={[
              styles.toolStatus,
              tool.status === 'failed' && styles.toolStatusFailed,
              tool.status === 'completed' && styles.toolStatusDone,
            ]}>{tool.status}</Text>
          </View>
          <Text selectable style={styles.toolDetail}>{tool.detail}</Text>
          {tool.result !== null && <Text numberOfLines={5} selectable style={styles.toolResult}>{tool.result}</Text>}
        </View>
      ))}
    </ScrollView>
  );
}

function SettingsView({
  capabilities,
  host,
  onRemove,
}: {
  capabilities: CapabilityProjection | null;
  host: NonNullable<ReturnType<typeof useRemoteSession>['host']>;
  onRemove(): void;
}) {
  return (
    <ScrollView contentContainerStyle={styles.scrollContent} style={styles.flex}>
      <Text style={styles.groupTitle}>Security</Text>
      <SettingRow label="Transport" value="Noise IK · pinned host key" />
      <SettingRow label="Approval gate" value="Strong biometrics" />
      <SettingRow label="Key protection" value="Device protected" />

      <Text style={[styles.groupTitle, styles.groupSpacing]}>Host</Text>
      <SettingRow label="Host ID" value={host.hostId} selectable />
      <SettingRow label="Relay" value={new URL(host.relayUrl).host} />
      <SettingRow label="Sequence" value={host.lastAppliedSequence.toString()} />

      <Text style={[styles.groupTitle, styles.groupSpacing]}>Session scope</Text>
      <SettingRow label="Workspace" value={capabilities?.workspace ?? 'Awaiting replay'} selectable />
      <SettingRow label="Model" value={capabilities?.modelId ?? 'Awaiting replay'} />
      <SettingRow label="File scope" value={capabilities?.fileScope ?? 'Awaiting replay'} />
      <SettingRow
        label="Shell"
        value={capabilities === null
          ? 'Awaiting replay'
          : capabilities.shell.enabled
            ? `${capabilities.shell.mode} · ${capabilities.shell.enforcedLayers.join(' + ')}`
            : 'Disabled'}
      />
      <SettingRow
        label="Camelid network tools"
        value={capabilities === null
          ? 'Awaiting replay'
          : capabilities.camelidNetworkTools ? 'Enabled' : 'Disabled'}
      />
      <SettingRow label="Tools" value={capabilities?.tools.join(', ') ?? 'Awaiting replay'} />

      <Pressable onPress={onRemove} style={({ pressed }) => [styles.removeButton, pressed && styles.pressed]}>
        <SymbolView
          name={{ ios: 'trash', android: 'delete', web: 'delete' }}
          size={18}
          tintColor={palette.danger}
        />
        <Text style={styles.removeLabel}>Remove from this device</Text>
      </Pressable>
    </ScrollView>
  );
}

function ApprovalModal({
  approval,
  capabilities,
  connected,
  connection,
  hostLabel,
  pendingAction,
  onClose,
  onDecision,
}: {
  approval: ApprovalProjection | null;
  capabilities: CapabilityProjection | null;
  connected: boolean;
  connection: string;
  hostLabel: string;
  pendingAction: boolean;
  onClose(): void;
  onDecision(approval: ApprovalProjection, decision: 'allow_once' | 'deny' | 'abort_turn'): Promise<void>;
}) {
  const record = approval === null ? null : asRecord(approval.record);
  const action = asRecord(record?.action);
  const actionable = approval?.status === 'pending' && connected && !pendingAction;
  return (
    <Modal animationType="slide" onRequestClose={onClose} presentationStyle="pageSheet" visible={approval !== null}>
      <SafeAreaView style={styles.modalSafeArea}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Review action</Text>
          <Pressable accessibilityLabel="Close approval" onPress={onClose} style={styles.iconButton}>
            <SymbolView
              name={{ ios: 'xmark', android: 'close', web: 'close' }}
              size={18}
              tintColor={palette.ink}
            />
          </Pressable>
        </View>
        <ScrollView contentContainerStyle={styles.modalContent}>
          <DetailRow label="Host" value={`${hostLabel} · ${statusLabel(connection)}`} />
          <DetailRow label="Workspace" value={capabilities?.workspace ?? 'Unavailable'} selectable />
          <DetailRow label="Model" value={capabilities?.modelId ?? 'Unavailable'} />
          <DetailRow label="Tool" value={string(record?.tool)} />
          <DetailRow label="Risk" value={string(record?.risk)} />
          <DetailRow label="Action" value={string(action?.kind)} />
          <DetailRow label="Details" value={string(record?.detail)} selectable />
          <DetailRow label="Digest" value={approval?.actionDigest ?? ''} selectable />
          <View style={styles.exactAction}>
            <Text style={styles.exactActionLabel}>EXACT ACTION RECORD</Text>
            <Text selectable style={styles.exactActionText}>{formatAction(action)}</Text>
          </View>
        </ScrollView>
        <View style={styles.modalActions}>
          <Pressable
            disabled={!actionable}
            onPress={() => approval !== null && void onDecision(approval, 'allow_once')}
            style={[styles.allowButton, !actionable && styles.disabledAction]}>
            {pendingAction ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.allowLabel}>Allow once</Text>}
          </Pressable>
          <View style={styles.actionPair}>
            <Pressable
              disabled={!actionable}
              onPress={() => approval !== null && void onDecision(approval, 'deny')}
              style={[styles.denyButton, !actionable && styles.disabledAction]}>
              <Text style={styles.denyLabel}>Deny</Text>
            </Pressable>
            <Pressable
              disabled={!actionable}
              onPress={() => approval !== null && void onDecision(approval, 'abort_turn')}
              style={[styles.abortButton, !actionable && styles.disabledAction]}>
              <Text style={styles.abortLabel}>Abort turn</Text>
            </Pressable>
          </View>
        </View>
      </SafeAreaView>
    </Modal>
  );
}

function ConnectionStatus({ connection, status }: { connection: string; status: string }) {
  const connected = connection === 'connected';
  return (
    <View style={styles.connectionRow}>
      <View style={[styles.connectionDot, connected && styles.connectionDotReady]} />
      <Text style={styles.connectionText}>{connection === 'connecting' ? 'Connecting' : connected ? statusLabel(status) : 'Offline'}</Text>
    </View>
  );
}

function SettingRow({ label, value, selectable = false }: { label: string; value: string; selectable?: boolean }) {
  return (
    <View style={styles.settingRow}>
      <Text style={styles.settingLabel}>{label}</Text>
      <Text numberOfLines={selectable ? undefined : 2} selectable={selectable} style={styles.settingValue}>{value}</Text>
    </View>
  );
}

function DetailRow({ label, value, selectable = false }: { label: string; value: string; selectable?: boolean }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text selectable={selectable} style={styles.detailValue}>{value || '—'}</Text>
    </View>
  );
}

function approvalSummary(approval: ApprovalProjection): string {
  const record = asRecord(approval.record);
  return `${string(record?.tool) || 'Action'} · ${string(record?.risk) || 'approval'}`;
}

function formatAction(action: Readonly<Record<string, unknown>> | null): string {
  if (action === null) return 'Unavailable';
  return JSON.stringify(action, null, 2);
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : null;
}

function string(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function statusLabel(value: string): string {
  return value.replace('_', ' ').replace(/^./, (character) => character.toUpperCase());
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Operation failed.';
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  safeArea: { backgroundColor: palette.canvas, flex: 1 },
  topBar: {
    alignItems: 'center',
    borderBottomColor: palette.line,
    borderBottomWidth: 1,
    flexDirection: 'row',
    minHeight: 68,
    paddingHorizontal: 18,
  },
  hostHeading: { flex: 1, minWidth: 0 },
  hostName: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 17 },
  connectionRow: { alignItems: 'center', flexDirection: 'row', gap: 6, marginTop: 3 },
  connectionDot: { backgroundColor: palette.secondary, borderRadius: 3, height: 6, width: 6 },
  connectionDotReady: { backgroundColor: palette.accent },
  connectionText: { color: palette.secondary, fontFamily: fonts.body, fontSize: 12 },
  iconButton: { alignItems: 'center', height: 40, justifyContent: 'center', width: 40 },
  pressed: { opacity: 0.58 },
  tabs: { borderBottomColor: palette.line, borderBottomWidth: 1, flexDirection: 'row', paddingHorizontal: 18 },
  tab: { alignItems: 'center', flex: 1, justifyContent: 'center', minHeight: 44 },
  tabSelected: { borderBottomColor: palette.accent, borderBottomWidth: 2 },
  tabLabel: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 13 },
  tabLabelSelected: { color: palette.ink, fontFamily: fonts.semibold },
  scrollContent: { paddingBottom: 110, paddingHorizontal: 18, paddingTop: 18 },
  approvalRow: {
    alignItems: 'center',
    backgroundColor: palette.warningSurface,
    borderColor: '#E8C89D',
    borderRadius: 6,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 11,
    marginBottom: 14,
    minHeight: 62,
    paddingHorizontal: 13,
  },
  approvalMark: { backgroundColor: palette.warning, borderRadius: 4, height: 8, width: 8 },
  approvalTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 14 },
  approvalSummary: { color: palette.secondary, fontFamily: fonts.body, fontSize: 12, marginTop: 2 },
  emptyTranscript: { borderTopColor: palette.line, borderTopWidth: 1, paddingTop: 44 },
  emptyTranscriptTitle: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 14, textAlign: 'center' },
  message: { borderBottomColor: palette.line, borderBottomWidth: 1, paddingBottom: 18, paddingTop: 16 },
  messageRole: { color: palette.secondary, fontFamily: fonts.semibold, fontSize: 10 },
  messageText: { color: palette.ink, fontFamily: fonts.body, fontSize: 15, lineHeight: 23, marginTop: 7 },
  groupTitle: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 14, marginBottom: 8 },
  centerState: { alignItems: 'center', justifyContent: 'center', minHeight: 140 },
  historyIntro: { color: palette.secondary, fontFamily: fonts.body, fontSize: 13, lineHeight: 19, marginBottom: 14 },
  historyHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  newSessionButton: { alignItems: 'center', borderColor: palette.line, borderRadius: 5, borderWidth: 1, flexDirection: 'row', gap: 5, minHeight: 34, paddingHorizontal: 10 },
  newSessionLabel: { color: palette.accent, fontFamily: fonts.semibold, fontSize: 12 },
  historyRow: { alignItems: 'center', borderBottomColor: palette.line, borderBottomWidth: 1, flexDirection: 'row', gap: 10, minHeight: 72, paddingVertical: 12 },
  historyRowSelected: { backgroundColor: palette.quiet },
  historyTitleRow: { alignItems: 'center', flexDirection: 'row', gap: 8 },
  historyTitle: { color: palette.ink, flex: 1, fontFamily: fonts.semibold, fontSize: 14 },
  historyMeta: { color: palette.secondary, fontFamily: fonts.body, fontSize: 12, marginTop: 4 },
  continueButton: { alignItems: 'center', alignSelf: 'flex-start', borderColor: palette.line, borderRadius: 4, borderWidth: 1, justifyContent: 'center', marginTop: 8, minHeight: 32, paddingHorizontal: 10 },
  continueLabel: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 12 },
  activeBadge: { backgroundColor: palette.quiet, borderRadius: 4, color: palette.accent, fontFamily: fonts.semibold, fontSize: 10, paddingHorizontal: 6, paddingVertical: 3 },
  historyFooter: { alignItems: 'center', backgroundColor: palette.quiet, borderTopColor: palette.line, borderTopWidth: 1, minHeight: 48, justifyContent: 'center', paddingHorizontal: 18 },
  historyFooterText: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 12 },
  groupSpacing: { marginTop: 28 },
  muted: { color: palette.secondary, fontFamily: fonts.body, fontSize: 13, paddingVertical: 12 },
  activityRow: { alignItems: 'flex-start', borderBottomColor: palette.line, borderBottomWidth: 1, flexDirection: 'row', gap: 10, paddingVertical: 12 },
  activityText: { color: palette.ink, flex: 1, fontFamily: fonts.body, fontSize: 14, lineHeight: 20 },
  toolRow: { borderBottomColor: palette.line, borderBottomWidth: 1, gap: 7, paddingVertical: 14 },
  toolHeader: { alignItems: 'center', flexDirection: 'row', justifyContent: 'space-between' },
  toolName: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 13 },
  toolStatus: { color: palette.warning, fontFamily: fonts.medium, fontSize: 11 },
  toolStatusDone: { color: palette.accent },
  toolStatusFailed: { color: palette.danger },
  toolDetail: { color: palette.secondary, fontFamily: fonts.body, fontSize: 12, lineHeight: 18 },
  toolResult: { backgroundColor: palette.quiet, borderRadius: 4, color: palette.ink, fontFamily: fonts.body, fontSize: 12, lineHeight: 18, padding: 10 },
  settingRow: { borderBottomColor: palette.line, borderBottomWidth: 1, gap: 5, paddingVertical: 13 },
  settingLabel: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 11 },
  settingValue: { color: palette.ink, fontFamily: fonts.body, fontSize: 13, lineHeight: 18 },
  removeButton: { alignItems: 'center', borderColor: '#DFC4C1', borderRadius: 6, borderWidth: 1, flexDirection: 'row', gap: 9, justifyContent: 'center', marginTop: 32, minHeight: 48 },
  removeLabel: { color: palette.danger, fontFamily: fonts.semibold, fontSize: 14 },
  errorBand: { backgroundColor: '#F8E9E7', borderTopColor: '#E7C3BF', borderTopWidth: 1, paddingHorizontal: 18, paddingVertical: 10 },
  errorText: { color: palette.danger, fontFamily: fonts.medium, fontSize: 12, lineHeight: 17 },
  commandNotice: { backgroundColor: palette.quiet, color: palette.secondary, fontFamily: fonts.body, fontSize: 12, paddingHorizontal: 18, paddingVertical: 8 },
  composer: { backgroundColor: palette.canvas, borderTopColor: palette.line, borderTopWidth: 1, paddingBottom: 12, paddingHorizontal: 14, paddingTop: 10 },
  composeRow: { alignItems: 'flex-end', flexDirection: 'row', gap: 9 },
  composeInput: { backgroundColor: palette.surface, borderColor: palette.line, borderRadius: 6, borderWidth: 1, color: palette.ink, flex: 1, fontFamily: fonts.body, fontSize: 14, maxHeight: 120, minHeight: 46, paddingHorizontal: 12, paddingVertical: 11 },
  sendButton: { alignItems: 'center', backgroundColor: palette.accent, borderRadius: 6, height: 46, justifyContent: 'center', width: 46 },
  sendDisabled: { backgroundColor: '#AAB3AE' },
  primaryPressed: { backgroundColor: palette.accentPressed },
  cancelButton: { alignItems: 'center', borderColor: '#DFC4C1', borderRadius: 6, borderWidth: 1, flexDirection: 'row', gap: 8, justifyContent: 'center', minHeight: 46 },
  cancelLabel: { color: palette.danger, fontFamily: fonts.semibold, fontSize: 14 },
  modalSafeArea: { backgroundColor: palette.canvas, flex: 1 },
  modalHeader: { alignItems: 'center', borderBottomColor: palette.line, borderBottomWidth: 1, flexDirection: 'row', minHeight: 62, paddingLeft: 18, paddingRight: 8 },
  modalTitle: { color: palette.ink, flex: 1, fontFamily: fonts.semibold, fontSize: 18 },
  modalContent: { padding: 18, paddingBottom: 120 },
  detailRow: { borderBottomColor: palette.line, borderBottomWidth: 1, gap: 5, paddingVertical: 12 },
  detailLabel: { color: palette.secondary, fontFamily: fonts.medium, fontSize: 11 },
  detailValue: { color: palette.ink, fontFamily: fonts.body, fontSize: 13, lineHeight: 19 },
  exactAction: { marginTop: 24 },
  exactActionLabel: { color: palette.secondary, fontFamily: fonts.semibold, fontSize: 10, marginBottom: 8 },
  exactActionText: { backgroundColor: palette.quiet, borderRadius: 6, color: palette.ink, fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }), fontSize: 12, lineHeight: 18, padding: 13 },
  modalActions: { backgroundColor: palette.canvas, borderTopColor: palette.line, borderTopWidth: 1, bottom: 0, gap: 9, left: 0, padding: 14, position: 'absolute', right: 0 },
  allowButton: { alignItems: 'center', backgroundColor: palette.accent, borderRadius: 6, justifyContent: 'center', minHeight: 48 },
  allowLabel: { color: '#FFFFFF', fontFamily: fonts.semibold, fontSize: 14 },
  actionPair: { flexDirection: 'row', gap: 9 },
  denyButton: { alignItems: 'center', borderColor: palette.line, borderRadius: 6, borderWidth: 1, flex: 1, justifyContent: 'center', minHeight: 46 },
  denyLabel: { color: palette.ink, fontFamily: fonts.semibold, fontSize: 14 },
  abortButton: { alignItems: 'center', borderColor: '#DFC4C1', borderRadius: 6, borderWidth: 1, flex: 1, justifyContent: 'center', minHeight: 46 },
  abortLabel: { color: palette.danger, fontFamily: fonts.semibold, fontSize: 14 },
  disabledAction: { opacity: 0.45 },
});
