import { initialSessionProjection, type RemoteEvent } from './reducer';
import { reconcileRemoteEvents } from './sessionReconciliation';

function event(sequence: number, type = 'future.observation'): RemoteEvent {
  return { sequence, type, payload: {} };
}

describe('remote session reconciliation', () => {
  test('buffers a live gap and applies it only after replay fills the missing sequence', () => {
    const initial = initialSessionProjection();
    const gap = reconcileRemoteEvents(initial, new Map(), [event(3)]);
    expect(gap.state.nextSequence).toBe(1);
    expect(gap.replayAfter).toBe(0);
    expect([...gap.buffered.keys()]).toEqual([3]);

    const filled = reconcileRemoteEvents(gap.state, gap.buffered, [event(1), event(2)]);
    expect(filled.state.nextSequence).toBe(4);
    expect(filled.replayAfter).toBeNull();
    expect(filled.buffered.size).toBe(0);
  });

  test('ignores duplicates and applies a contiguous batch in order', () => {
    const first = reconcileRemoteEvents(initialSessionProjection(), new Map(), [
      event(1),
      event(1),
      event(2),
    ]);
    expect(first.state.nextSequence).toBe(3);
    expect(first.buffered.size).toBe(0);

    const duplicate = reconcileRemoteEvents(first.state, first.buffered, [event(1), event(2)]);
    expect(duplicate.state).toEqual(first.state);
    expect(duplicate.buffered.size).toBe(0);
  });

  test('preserves the first event received for one sequence', () => {
    const pending = reconcileRemoteEvents(initialSessionProjection(), new Map(), [
      event(2, 'first'),
      event(2, 'replacement'),
    ]);
    expect(pending.buffered.get(2)?.type).toBe('first');
  });
});
