import type { StoredHost } from './hostStore';
import type { RemoteEnvelope, SessionCatalog } from './protocol';

export function messageTargetsSelectedSession(
  message: RemoteEnvelope,
  selectedSessionId: string | null,
): boolean {
  return selectedSessionId !== null && message.session_id === selectedSessionId;
}

export function canControlSelectedSession(
  host: StoredHost | null,
  catalog: SessionCatalog | null,
  selectedSessionId: string | null,
): boolean {
  if (host === null || selectedSessionId === null) return false;
  const activeSessionId = catalog?.activeSessionId ?? host.sessionId;
  return selectedSessionId === activeSessionId;
}

export function mergeSessionCatalogPage(
  current: SessionCatalog | null,
  page: SessionCatalog,
  expectedPaginationRevision: string | null,
): SessionCatalog {
  const base = expectedPaginationRevision === null ? null : current;
  if (
    expectedPaginationRevision !== null &&
    (base === null || base.revision !== expectedPaginationRevision || page.revision !== expectedPaginationRevision)
  ) {
    throw new Error('Remote session catalog changed during pagination. Refresh history.');
  }
  const sessions = base === null ? [...page.sessions] : [...base.sessions, ...page.sessions];
  if (new Set(sessions.map((entry) => entry.historyId)).size !== sessions.length) {
    throw new Error('Remote session catalog repeated a history entry.');
  }
  return { ...page, sessions };
}
