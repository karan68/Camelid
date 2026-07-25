import { applyRemoteEvent, type RemoteEvent, type SessionProjection } from './reducer';

export interface ReconciliationResult {
  state: SessionProjection;
  buffered: Map<number, RemoteEvent>;
  replayAfter: number | null;
}

export function reconcileRemoteEvents(
  state: SessionProjection,
  buffered: ReadonlyMap<number, RemoteEvent>,
  incoming: readonly RemoteEvent[],
): ReconciliationResult {
  const pending = new Map(buffered);
  for (const event of incoming) {
    if (event.sequence >= state.nextSequence && !pending.has(event.sequence)) {
      pending.set(event.sequence, event);
    }
  }

  let next = state;
  while (true) {
    const event = pending.get(next.nextSequence);
    if (event === undefined) break;
    const result = applyRemoteEvent(next, event);
    if (result.kind !== 'applied') break;
    pending.delete(event.sequence);
    next = result.state;
  }

  for (const sequence of pending.keys()) {
    if (sequence < next.nextSequence) pending.delete(sequence);
  }
  const firstPending = pending.size === 0 ? null : Math.min(...pending.keys());
  return {
    state: next,
    buffered: pending,
    replayAfter: firstPending !== null && firstPending > next.nextSequence
      ? next.nextSequence - 1
      : null,
  };
}
