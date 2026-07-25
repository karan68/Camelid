import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState } from 'react-native';

import CamelidRemoteCrypto from '../../modules/camelid-remote-crypto';
import { authorizeNativeApproval } from './nativeApprovalGate';
import { activateSession, approvalDecision, ApprovalDecision, cancelTurn, createSession, replayRequest, sessionCatalogRequest, startTurn } from './commands';
import { createSecureHostStore } from './secureHostStore';
import type { StoredHost } from './hostStore';
import { nativeSha256, randomUuid } from './nativeCrypto';
import {
  parseCommandResult,
  parseEventBatch,
  parseReplayEnd,
  parseSessionCatalog,
  type SessionCatalog,
  type RemoteEnvelope,
} from './protocol';
import {
  applySessionSnapshot,
  initialSessionProjection,
  type ApprovalProjection,
  type SessionProjection,
} from './reducer';
import { connectSession, type SessionTransport } from './sessionTransport';
import { reconcileRemoteEvents } from './sessionReconciliation';
import { canControlSelectedSession, mergeSessionCatalogPage, messageTargetsSelectedSession } from './sessionSelection';

export type ConnectionState = 'connecting' | 'connected' | 'offline';

export interface RemoteSessionController {
  host: StoredHost | null;
  projection: SessionProjection;
  connection: ConnectionState;
  error: string | null;
  commandNotice: string | null;
  catalog: SessionCatalog | null;
  selectedSessionId: string | null;
  canControl: boolean;
  connect(): Promise<void>;
  selectSession(sessionId: string): Promise<void>;
  createSession(): Promise<void>;
  activateSession(sessionId: string): Promise<void>;
  start(text: string): Promise<void>;
  cancel(): Promise<void>;
  decide(approvalId: string, decision: ApprovalDecision): Promise<void>;
}

export function useRemoteSession(hostId: string | undefined): RemoteSessionController {
  const [host, setHost] = useState<StoredHost | null>(null);
  const [projection, setProjection] = useState(initialSessionProjection);
  const [connection, setConnection] = useState<ConnectionState>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [commandNotice, setCommandNotice] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<SessionCatalog | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const projectionRef = useRef(projection);
  const catalogRef = useRef<SessionCatalog | null>(null);
  const transportRef = useRef<SessionTransport | null>(null);
  const connectingRef = useRef(false);
  const generationRef = useRef(0);
  const activeRef = useRef(AppState.currentState === 'active');
  const selectedSessionRef = useRef<string | null>(null);
  const pendingActiveSessionRef = useRef<string | null>(null);
  const expectedCatalogRevisionRef = useRef<string | null>(null);

  const commitProjection = useCallback((next: SessionProjection) => {
    projectionRef.current = next;
    setProjection(next);
  }, []);

  const commitCatalog = useCallback((next: SessionCatalog | null) => {
    catalogRef.current = next;
    setCatalog(next);
  }, []);

  const connect = useCallback(async () => {
    if (hostId === undefined || connectingRef.current) return;
    connectingRef.current = true;
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    setConnection('connecting');
    setError(null);
    setCommandNotice(null);
    let pendingReplayAfter: number | null = null;
    const pendingCatalogPage: {
      current: {
        cursor: NonNullable<SessionCatalog['nextCursor']>;
        revision: string;
      } | null;
    } = { current: null };
    let lastReplayAfter: number | null = null;
    let bufferedEvents = new Map<number, ReturnType<typeof parseEventBatch>[number]>();
    try {
      await transportRef.current?.close();
      transportRef.current = null;
      const store = await createSecureHostStore();
      const selected = (await store.list()).find((candidate) => candidate.hostId === hostId);
      if (selected === undefined) throw new Error('Paired host was not found.');
      setHost(selected);
      selectedSessionRef.current = selected.sessionId;
      expectedCatalogRevisionRef.current = null;
      setSelectedSessionId(selected.sessionId);
      commitCatalog(null);
      commitProjection({
        ...initialSessionProjection(),
        nextSequence: 1,
      });

      const requestReplay = async (afterSequence: number) => {
        if (lastReplayAfter === afterSequence) return;
        const transport = transportRef.current;
        if (transport === null) {
          pendingReplayAfter = afterSequence;
          return;
        }
        const messageId = randomUuid();
        const replaySessionId = selectedSessionRef.current ?? selected.sessionId;
        await transport.send(
          messageId,
          replayRequest({
            ...scope(selected, messageId),
            sessionId: replaySessionId,
          }, afterSequence),
        );
        lastReplayAfter = afterSequence;
      };

      const requestCatalogPage = async (
        cursor: NonNullable<SessionCatalog['nextCursor']>,
        revision: string,
      ) => {
        const transport = transportRef.current;
        if (transport === null) {
          pendingCatalogPage.current = { cursor, revision };
          return;
        }
        expectedCatalogRevisionRef.current = revision;
        const messageId = randomUuid();
        await transport.send(
          messageId,
          sessionCatalogRequest({
            messageId,
            hostId: selected.hostId,
            deviceId: selected.deviceId,
            sentAtUnixMs: Date.now(),
          }, { cursor, revision }),
        );
      };

      const onMessage = async (message: RemoteEnvelope) => {
        requireEnvelopeHost(message, selected);
        if (message.kind === 'session_catalog') {
          const page = parseSessionCatalog(message);
          const activeSessionChanged = page.activeSessionId !== selected.sessionId;
          if (activeSessionChanged) {
            const previousActiveSessionId = selected.sessionId;
            const followNewActive =
              selectedSessionRef.current === previousActiveSessionId ||
              pendingActiveSessionRef.current === page.activeSessionId;
            const updatedHost = await store.updateActiveSession(selected.hostId, page.activeSessionId);
            Object.assign(selected, updatedHost);
            setHost(updatedHost);
            pendingActiveSessionRef.current = null;
            if (followNewActive) {
              selectedSessionRef.current = page.activeSessionId;
              setSelectedSessionId(page.activeSessionId);
              commitProjection(initialSessionProjection());
              lastReplayAfter = null;
              await requestReplay(0);
            }
          }
          const expectedRevision = expectedCatalogRevisionRef.current;
          expectedCatalogRevisionRef.current = null;
          const nextCatalog = mergeSessionCatalogPage(
            activeSessionChanged ? null : catalogRef.current,
            page,
            expectedRevision,
          );
          commitCatalog(nextCatalog);
          if (page.nextCursor !== null) {
            await requestCatalogPage(page.nextCursor, page.revision);
          }
          return;
        }
        if (message.kind === 'event_batch') {
          if (!messageTargetsSelectedSession(message, selectedSessionRef.current)) return;
          const previousSequence = projectionRef.current.nextSequence;
          const reconciliation = reconcileRemoteEvents(
            projectionRef.current,
            bufferedEvents,
            parseEventBatch(message),
          );
          const next = reconciliation.state;
          bufferedEvents = reconciliation.buffered;
          commitProjection(next);
          if (next.nextSequence !== previousSequence) {
            lastReplayAfter = null;
            const replaySessionId = message.session_id;
            if (replaySessionId === null) throw new Error('Replay event session identity is missing.');
            const updatedHost = await store.updateLastAppliedSequence(
              selected.hostId,
              replaySessionId,
              next.nextSequence - 1,
            );
            Object.assign(selected, updatedHost);
            setHost(updatedHost);
          }
          if (reconciliation.replayAfter !== null) {
            await requestReplay(reconciliation.replayAfter);
          }
          return;
        }
        if (message.kind === 'replay_end') {
          if (!messageTargetsSelectedSession(message, selectedSessionRef.current)) return;
          const replay = parseReplayEnd(message);
          const next = applySessionSnapshot(projectionRef.current, replay.sessionState);
          commitProjection(next);
          if (replay.hasMore) {
            lastReplayAfter = null;
            await requestReplay(next.nextSequence - 1);
          } else if (replay.lastSequence > next.nextSequence - 1) {
            lastReplayAfter = null;
            await requestReplay(next.nextSequence - 1);
          } else if (replay.lastSequence < next.nextSequence - 1) {
            throw new Error('Replay head moved behind the applied event cursor.');
          }
          return;
        }
        if (message.kind === 'command_result') {
          if (message.session_id !== selected.sessionId) return;
          const result = parseCommandResult(message);
          setCommandNotice(result.message);
          if (result.status === 'rejected') {
            pendingActiveSessionRef.current = null;
            setError(result.message);
          }
          return;
        }
        if (message.kind !== 'pong') {
          throw new Error(`Unexpected remote message kind: ${message.kind}.`);
        }
      };

      const transport = await connectSession({
        host: selected,
        crypto: CamelidRemoteCrypto,
        sha256: nativeSha256,
        messageId: randomUuid,
        onMessage,
        onClose: (reason) => {
          if (generationRef.current !== generation) return;
          transportRef.current = null;
          setConnection('offline');
          commitProjection({ ...projectionRef.current, status: 'offline' });
          if (reason !== null) setError(reason.message);
        },
        replayAfterSequence: 0,
      });
      if (generationRef.current !== generation) {
        await transport.close();
        return;
      }
      transportRef.current = transport;
      setConnection('connected');
      if (pendingReplayAfter !== null) {
        const after = pendingReplayAfter;
        pendingReplayAfter = null;
        await requestReplay(after);
      }
      if (pendingCatalogPage.current !== null) {
        const pending = pendingCatalogPage.current;
        pendingCatalogPage.current = null;
        await requestCatalogPage(pending.cursor, pending.revision);
      }
    } catch (caught) {
      if (generationRef.current === generation) {
        setConnection('offline');
        setError(asMessage(caught));
        commitProjection({ ...projectionRef.current, status: 'offline' });
      }
    } finally {
      connectingRef.current = false;
      if (
        activeRef.current &&
        generationRef.current !== generation &&
        transportRef.current === null
      ) {
        setReconnectAttempt((attempt) => attempt + 1);
      }
    }
  }, [commitCatalog, commitProjection, hostId]);

  useEffect(() => {
    const start = setTimeout(() => void connect(), 0);
    return () => {
      clearTimeout(start);
      generationRef.current += 1;
      const transport = transportRef.current;
      transportRef.current = null;
      if (transport !== null) void transport.close();
    };
  }, [connect, reconnectAttempt]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextState) => {
      activeRef.current = nextState === 'active';
      if (nextState === 'active') {
        setTimeout(() => void connect(), 0);
        return;
      }
      generationRef.current += 1;
      const transport = transportRef.current;
      transportRef.current = null;
      if (transport !== null) void transport.close();
      setConnection('offline');
      commitProjection({ ...projectionRef.current, status: 'offline' });
    });
    return () => subscription.remove();
  }, [commitProjection, connect]);

  const sendCommand = useCallback(async (
    builder: (messageId: string, commandId: string) => string,
  ) => {
    if (host === null || transportRef.current === null || connection !== 'connected') {
      throw new Error('Remote host is not connected.');
    }
    const messageId = randomUuid();
    const commandId = randomUuid();
    setError(null);
    setCommandNotice(null);
    await transportRef.current.send(messageId, builder(messageId, commandId));
  }, [connection, host]);

  const selectSession = useCallback(async (sessionId: string) => {
    if (host === null || transportRef.current === null || connection !== 'connected') {
      throw new Error('Remote host is not connected.');
    }
    const summary = catalog?.sessions.find((entry) => entry.historyId === sessionId);
    if (summary === undefined) throw new Error('Session history is not in the current catalog.');
    selectedSessionRef.current = sessionId;
    setSelectedSessionId(sessionId);
    commitProjection(initialSessionProjection());
    setError(null);
    setCommandNotice(null);
    const messageId = randomUuid();
    await transportRef.current.send(
      messageId,
      replayRequest({
        messageId,
        hostId: host.hostId,
        deviceId: host.deviceId,
        sessionId,
        sentAtUnixMs: Date.now(),
      }, 0),
    );
  }, [catalog?.sessions, commitProjection, connection, host]);

  const createNewSession = useCallback(async () => {
    if (host === null || transportRef.current === null || connection !== 'connected') {
      throw new Error('Remote host is not connected.');
    }
    if (!canControlSelectedSession(host, catalog, selectedSessionId)) {
      throw new Error('Return to the active session before creating a new session.');
    }
    const sessionId = randomUuid();
    const messageId = randomUuid();
    const commandId = randomUuid();
    pendingActiveSessionRef.current = sessionId;
    try {
      await transportRef.current.send(messageId, createSession({
        ...scope(host, messageId),
        commandId,
      }, sessionId));
    } catch (caught) {
      pendingActiveSessionRef.current = null;
      throw caught;
    }
  }, [catalog, connection, host, selectedSessionId]);

  const activateHistory = useCallback(async (sessionId: string) => {
    if (host === null || transportRef.current === null || connection !== 'connected') {
      throw new Error('Remote host is not connected.');
    }
    const summary = catalog?.sessions.find((entry) => entry.historyId === sessionId);
    if (summary === undefined || !summary.continuable || summary.source !== 'remote') {
      throw new Error('This history cannot be continued with the current host identity.');
    }
    const messageId = randomUuid();
    const commandId = randomUuid();
    pendingActiveSessionRef.current = sessionId;
    try {
      await transportRef.current.send(messageId, activateSession({
        ...scope(host, messageId),
        commandId,
      }, sessionId));
    } catch (caught) {
      pendingActiveSessionRef.current = null;
      throw caught;
    }
  }, [catalog?.sessions, connection, host]);

  const start = useCallback(async (text: string) => {
    if (host === null) throw new Error('Remote host is unavailable.');
    const turnId = randomUuid();
    await sendCommand((messageId, commandId) => startTurn({
      ...scope(host, messageId),
      commandId,
      turnId,
    }, text));
  }, [host, sendCommand]);

  const cancel = useCallback(async () => {
    if (host === null || projectionRef.current.currentTurnId === null) {
      throw new Error('There is no active turn to cancel.');
    }
    const turnId = projectionRef.current.currentTurnId;
    await sendCommand((messageId, commandId) => cancelTurn({
      ...scope(host, messageId),
      commandId,
      turnId,
    }));
  }, [host, sendCommand]);

  const decide = useCallback(async (approvalId: string, decision: ApprovalDecision) => {
    if (host === null) throw new Error('Remote host is unavailable.');
    const approval = pendingApproval(projectionRef.current, approvalId);
    if (decision === 'allow_once') {
      const gate = await authorizeNativeApproval(true);
      if (!gate.authorized) throw new Error('Biometric approval was not completed.');
    }
    await sendCommand((messageId, commandId) => approvalDecision({
      ...scope(host, messageId),
      commandId,
      turnId: approval.turnId,
    }, {
      callId: approval.callId,
      approvalId: approval.approvalId,
      actionDigest: approval.actionDigest,
      decision,
    }));
  }, [host, sendCommand]);

  return {
    host,
    projection,
    connection,
    error,
    commandNotice,
    catalog,
    selectedSessionId,
    canControl: canControlSelectedSession(host, catalog, selectedSessionId),
    connect,
    selectSession,
    createSession: createNewSession,
    activateSession: activateHistory,
    start,
    cancel,
    decide,
  };
}


function scope(host: StoredHost, messageId: string) {
  return {
    messageId,
    hostId: host.hostId,
    deviceId: host.deviceId,
    sessionId: host.sessionId,
    sentAtUnixMs: Date.now(),
  };
}

function requireEnvelopeHost(message: RemoteEnvelope, host: StoredHost): void {
  if (
    message.host_id !== host.hostId ||
    message.device_id !== host.deviceId
  ) {
    throw new Error('Remote message identity changed.');
  }
}

function pendingApproval(state: SessionProjection, approvalId: string): ApprovalProjection {
  const approval = state.approvals[approvalId];
  if (approval === undefined || approval.status !== 'pending') {
    throw new Error('Approval is no longer pending.');
  }
  return approval;
}

function asMessage(value: unknown): string {
  return value instanceof Error ? value.message : 'Remote session failed.';
}
