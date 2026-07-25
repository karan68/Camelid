import {
  applyRemoteEvent,
  applySessionSnapshot,
  initialSessionProjection,
  RemoteEvent,
  SessionProjection,
} from './reducer';

function event(
  sequence: number,
  type: string,
  payload: Record<string, unknown> = {},
  turnId: string | null = null,
): RemoteEvent {
  return { sequence, type, payload, turnId };
}

function apply(state: SessionProjection, next: RemoteEvent): SessionProjection {
  const result = applyRemoteEvent(state, next);
  expect(result.kind).toBe('applied');
  return result.state;
}

describe('remote replay reducer', () => {
  test('ignores duplicates and blocks gaps until replay fills them', () => {
    const initial = initialSessionProjection();
    const first = apply(initial, event(1, 'session.state_changed', { state: 'running' }));
    expect(applyRemoteEvent(first, event(1, 'turn.finished'))).toEqual({
      kind: 'duplicate',
      state: first,
    });

    const gap = applyRemoteEvent(first, event(3, 'turn.finished', { outcome: 'completed' }));
    expect(gap).toMatchObject({ kind: 'gap', expected: 2, received: 3 });
    expect(gap.state.status).toBe('running');
    expect(gap.state.gapAfterSequence).toBe(1);
  });

  test('advances unknown events without mutating privileged state', () => {
    const initial = { ...initialSessionProjection(), status: 'idle' as const };
    const result = applyRemoteEvent(
      initial,
      event(1, 'future.admin.grant', { approval: 'always', shell: true }),
    );
    expect(result.kind).toBe('applied');
    expect(result.state).toEqual({ ...initial, nextSequence: 2 });
  });

  test('keeps settled approvals non-actionable and ignores duplicate settlement', () => {
    let state = apply(
      initialSessionProjection(),
      event(1, 'approval.required', {
        approval_id: 'approval-1',
        record: {
          schema: 'camelid.approval-record/v1',
          call_id: 'call-1',
          action_digest: 'sha256:action',
          kind: 'write_file',
        },
      }, 'turn-1'),
    );
    expect(state.approvals['approval-1'].status).toBe('pending');

    state = apply(
      state,
      event(2, 'approval.settled', { approval_id: 'approval-1', decision: 'allow_once' }),
    );
    expect(state.approvals['approval-1'].status).toBe('allowed_once');
    const duplicate = applyRemoteEvent(
      state,
      event(2, 'approval.settled', { approval_id: 'approval-1', decision: 'deny' }),
    );
    expect(duplicate.kind).toBe('duplicate');
    expect(duplicate.state.approvals['approval-1'].status).toBe('allowed_once');
  });

  test('coalesces final answer and lets terminal state win over late deltas', () => {
    let state = apply(initialSessionProjection(), event(1, 'model.delta', { content: 'par' }));
    state = apply(state, event(2, 'model.delta', { content: 'tial' }));
    expect(state.streamingAnswer).toBe('partial');

    state = apply(state, event(3, 'model.answer', { content: 'complete' }));
    expect(state.streamingAnswer).toBe('');
    expect(state.transcript).toEqual([{ role: 'assistant', content: 'complete' }]);

    state = apply(state, event(4, 'turn.finished', { outcome: 'completed' }));
    state = apply(state, event(5, 'model.delta', { content: 'late' }));
    expect(state.status).toBe('completed');
    expect(state.streamingAnswer).toBe('');
  });

  test('does not create approval authority from malformed records', () => {
    const state = apply(
      initialSessionProjection(),
      event(1, 'approval.required', {
        approval_id: 'approval-1',
        record: { summary: 'missing digest' },
      }),
    );
    expect(state.approvals).toEqual({});
    expect(state.status).toBe('offline');
  });

  test('preserves the first approval digest when an ID is repeated at a later sequence', () => {
    let state = apply(
      initialSessionProjection(),
      event(1, 'approval.required', {
        approval_id: 'approval-1',
        record: {
          schema: 'camelid.approval-record/v1',
          call_id: 'call-1',
          action_digest: 'sha256:first',
          kind: 'write_file',
        },
      }, 'turn-1'),
    );
    state = apply(
      state,
      event(2, 'approval.required', {
        approval_id: 'approval-1',
        record: {
          schema: 'camelid.approval-record/v1',
          call_id: 'call-2',
          action_digest: 'sha256:replacement',
          kind: 'run_shell',
        },
      }, 'turn-1'),
    );
    expect(state.approvals['approval-1'].actionDigest).toBe('sha256:first');
    expect(state.status).toBe('failed');
    expect(state.protocolError).not.toBeNull();
  });

  test('keeps cancellation pending until an authoritative terminal event', () => {
    let state = apply(
      initialSessionProjection(),
      event(1, 'session.state_changed', { state: 'running' }),
    );
    state = apply(state, event(2, 'session.state_changed', { state: 'cancelling' }));
    state = apply(state, event(3, 'model.delta', { content: 'in flight' }));
    expect(state.status).toBe('cancelling');
    state = apply(state, event(4, 'turn.finished', { outcome: 'aborted' }));
    expect(state.status).toBe('aborted');
    expect(state.streamingAnswer).toBe('');
  });

  test('makes pending approval authority inert as soon as cancellation is durable', () => {
    let state = apply(
      initialSessionProjection(),
      event(1, 'approval.required', {
        approval_id: 'approval-1',
        record: {
          schema: 'camelid.approval-record/v1',
          call_id: 'call-1',
          action_digest: 'sha256:action',
        },
      }, 'turn-1'),
    );
    state = apply(state, event(2, 'session.state_changed', { state: 'cancelling' }, 'turn-1'));
    expect(state.approvals['approval-1'].status).toBe('aborted');
    state = apply(state, event(3, 'approval.settled', {
      approval_id: 'approval-1',
      decision: 'invalidated_by_cancel',
    }, 'turn-1'));
    expect(state.approvals['approval-1'].status).toBe('aborted');
  });

  test('ignores orphan settlements and repeated final answers while advancing sequence', () => {
    let state = apply(
      initialSessionProjection(),
      event(1, 'approval.settled', { approval_id: 'missing', decision: 'allow_once' }),
    );
    state = apply(state, event(2, 'model.answer', { content: 'first' }));
    state = apply(state, event(3, 'model.answer', { content: 'replacement' }));
    expect(state.nextSequence).toBe(4);
    expect(state.approvals).toEqual({});
    expect(state.transcript).toEqual([{ role: 'assistant', content: 'first' }]);
  });

  test('projects replay status, plan, and tool completion without parsing display detail', () => {
    let state = applySessionSnapshot(initialSessionProjection(), 'idle');
    expect(state.status).toBe('idle');
    state = apply(state, event(1, 'turn.accepted', { turn_id: 'turn-1' }, 'turn-1'));
    state = apply(state, event(2, 'plan.updated', {
      steps: [{ status: 'in_progress', text: 'Inspect the target' }],
    }, 'turn-1'));
    state = apply(state, event(3, 'tool.call', {
      call_id: 'call-1',
      tool: 'read_file',
      detail: 'read src/lib.rs',
    }, 'turn-1'));
    state = apply(state, event(4, 'tool.result', {
      call_id: 'call-1',
      tool: 'read_file',
      is_error: false,
      content: 'file content',
    }, 'turn-1'));
    expect(state.currentTurnId).toBe('turn-1');
    expect(state.status).toBe('running');
    expect(state.plan).toEqual([{ status: 'in_progress', text: 'Inspect the target' }]);
    expect(state.tools[0]).toMatchObject({ callId: 'call-1', status: 'completed', result: 'file content' });
  });

  test('projects the descriptive host capability snapshot for approval context', () => {
    const state = apply(initialSessionProjection(), event(1, 'host.capabilities', {
      workspace: 'C:\\work\\project',
      model_id: 'qwen-tool-row',
      model_artifact_sha256: 'sha256:model',
      tools: ['read_file', 'write_file'],
      file_scope: 'canonical_workspace',
      shell: {
        enabled: false,
        mode: 'disabled',
        enforced_layers: [],
        note: null,
      },
      camelid_network_tools: false,
    }));
    expect(state.capabilities).toMatchObject({
      workspace: 'C:\\work\\project',
      modelId: 'qwen-tool-row',
      tools: ['read_file', 'write_file'],
      shell: { enabled: false, mode: 'disabled' },
    });
  });
});
