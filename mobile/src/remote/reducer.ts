export type SessionStatus =
  | 'offline'
  | 'idle'
  | 'running'
  | 'waiting_approval'
  | 'cancelling'
  | 'completed'
  | 'aborted'
  | 'failed';

export type ApprovalStatus = 'pending' | 'allowed_once' | 'denied' | 'aborted' | 'expired';

export interface ApprovalProjection {
  approvalId: string;
  turnId: string;
  callId: string;
  actionDigest: string;
  status: ApprovalStatus;
  record: unknown;
}

export interface PlanProjection {
  status: 'pending' | 'in_progress' | 'done';
  text: string;
}

export interface ToolProjection {
  callId: string;
  tool: string;
  detail: string;
  status: 'running' | 'completed' | 'failed';
  result: string | null;
}

export interface CapabilityProjection {
  workspace: string;
  modelId: string;
  modelArtifactSha256: string;
  tools: readonly string[];
  fileScope: string;
  shell: {
    enabled: boolean;
    mode: string;
    enforcedLayers: readonly string[];
    note: string | null;
  };
  camelidNetworkTools: boolean;
}

export interface SessionProjection {
  nextSequence: number;
  status: SessionStatus;
  capabilities: CapabilityProjection | null;
  currentTurnId: string | null;
  transcript: readonly { role: 'user' | 'assistant'; content: string }[];
  streamingAnswer: string;
  plan: readonly PlanProjection[];
  tools: readonly ToolProjection[];
  approvals: Readonly<Record<string, ApprovalProjection>>;
  gapAfterSequence: number | null;
  protocolError: string | null;
}

export interface RemoteEvent {
  sequence: number;
  type: string;
  payload: Readonly<Record<string, unknown>>;
  turnId?: string | null;
}

export type ApplyResult =
  | { kind: 'applied'; state: SessionProjection }
  | { kind: 'duplicate'; state: SessionProjection }
  | { kind: 'gap'; state: SessionProjection; expected: number; received: number };

export const initialSessionProjection = (): SessionProjection => ({
  nextSequence: 1,
  status: 'offline',
  capabilities: null,
  currentTurnId: null,
  transcript: [],
  streamingAnswer: '',
  plan: [],
  tools: [],
  approvals: {},
  gapAfterSequence: null,
  protocolError: null,
});

export function applyRemoteEvent(state: SessionProjection, event: RemoteEvent): ApplyResult {
  if (!Number.isSafeInteger(event.sequence) || event.sequence < 1) {
    return { kind: 'gap', state, expected: state.nextSequence, received: event.sequence };
  }
  if (event.sequence < state.nextSequence) {
    return { kind: 'duplicate', state };
  }
  if (event.sequence > state.nextSequence) {
    return {
      kind: 'gap',
      state: { ...state, gapAfterSequence: state.nextSequence - 1 },
      expected: state.nextSequence,
      received: event.sequence,
    };
  }

  const advanced = reduceKnownEvent(state, event);
  return {
    kind: 'applied',
    state: {
      ...advanced,
      nextSequence: state.nextSequence + 1,
      gapAfterSequence: null,
    },
  };
}

export function applySessionSnapshot(
  state: SessionProjection,
  sessionState: 'armed' | 'idle' | 'running' | 'waiting_approval' | 'cancelling' | 'failed' | 'closed',
): SessionProjection {
  const status: SessionStatus =
    sessionState === 'armed' || sessionState === 'idle'
      ? 'idle'
      : sessionState === 'closed'
        ? 'offline'
        : sessionState;
  return { ...state, status };
}

function reduceKnownEvent(state: SessionProjection, event: RemoteEvent): SessionProjection {
  switch (event.type) {
    case 'host.capabilities': {
      const capabilities = readCapabilities(event.payload);
      return capabilities === null ? state : { ...state, capabilities };
    }
    case 'session.armed': {
      return { ...state, status: 'idle' };
    }
    case 'user.message': {
      const content = readString(event.payload, 'content');
      return content === null
        ? state
        : { ...state, transcript: [...state.transcript, { role: 'user', content }] };
    }
    case 'turn.accepted': {
      const turnId = readString(event.payload, 'turn_id');
      return turnId === null ? state : { ...state, currentTurnId: turnId, status: 'running' };
    }
    case 'model.delta': {
      if (isTerminal(state.status)) return state;
      const content = readString(event.payload, 'content');
      return content === null ? state : { ...state, streamingAnswer: state.streamingAnswer + content };
    }
    case 'model.answer': {
      const content = readString(event.payload, 'content');
      const lastMessage = state.transcript[state.transcript.length - 1];
      if (lastMessage?.role === 'assistant') return { ...state, streamingAnswer: '' };
      return content === null
        ? state
        : {
            ...state,
            streamingAnswer: '',
            transcript: [...state.transcript, { role: 'assistant', content }],
          };
    }
    case 'approval.required': {
      const approvalId = readString(event.payload, 'approval_id');
      const record = readRecord(event.payload, 'record');
      const actionDigest = record === null ? null : readString(record, 'action_digest');
      const callId = record === null ? null : readString(record, 'call_id');
      const schema = record === null ? null : readString(record, 'schema');
      if (
        approvalId === null ||
        event.turnId == null ||
        callId === null ||
        actionDigest === null ||
        schema !== 'camelid.approval-record/v1' ||
        record === null
      ) {
        return state;
      }
      const existing = state.approvals[approvalId];
      if (existing !== undefined) {
        return existing.actionDigest === actionDigest
          ? state
          : {
              ...state,
              status: 'failed',
              protocolError: 'Approval identity was reused with different authority.',
            };
      }
      return {
        ...state,
        status: 'waiting_approval',
        approvals: {
          ...state.approvals,
          [approvalId]: {
            approvalId,
            turnId: event.turnId,
            callId,
            actionDigest,
            status: 'pending',
            record,
          },
        },
      };
    }
    case 'plan.updated': {
      const steps = readPlan(event.payload.steps);
      return steps === null ? state : { ...state, plan: steps };
    }
    case 'tool.call': {
      const callId = readString(event.payload, 'call_id');
      const tool = readString(event.payload, 'tool');
      const detail = readString(event.payload, 'detail');
      if (callId === null || tool === null || detail === null) return state;
      return {
        ...state,
        tools: [...state.tools.filter((entry) => entry.callId !== callId), {
          callId,
          tool,
          detail,
          status: 'running',
          result: null,
        }],
      };
    }
    case 'tool.result': {
      const callId = readString(event.payload, 'call_id');
      const content = readString(event.payload, 'content');
      const isError = event.payload.is_error;
      if (callId === null || content === null || typeof isError !== 'boolean') return state;
      return {
        ...state,
        tools: state.tools.map((entry) => entry.callId === callId
          ? { ...entry, status: isError ? 'failed' : 'completed', result: content }
          : entry),
      };
    }
    case 'approval.settled':
    case 'approval.expired': {
      const approvalId = readString(event.payload, 'approval_id');
      if (approvalId === null || state.approvals[approvalId] === undefined) return state;
      const status = approvalStatus(event);
      if (status === null) return state;
      return {
        ...state,
        approvals: {
          ...state.approvals,
          [approvalId]: { ...state.approvals[approvalId], status },
        },
      };
    }
    case 'session.state_changed': {
      const status = readSessionStatus(event.payload, 'state');
      return status === null
        ? state
        : {
            ...state,
            status,
            approvals: status === 'cancelling'
              ? settlePendingApprovals(state.approvals, 'aborted')
              : state.approvals,
          };
    }
    case 'turn.finished': {
      const outcome = readString(event.payload, 'outcome');
      return {
        ...state,
        streamingAnswer: '',
        currentTurnId: null,
        approvals: settlePendingApprovals(state.approvals, outcome === 'completed' ? 'denied' : 'aborted'),
        status: outcome === 'completed' ? 'completed' : outcome === 'aborted' ? 'aborted' : 'failed',
      };
    }
    default:
      return state;
  }
}

function readCapabilities(value: Readonly<Record<string, unknown>>): CapabilityProjection | null {
  const workspace = readString(value, 'workspace');
  const modelId = readString(value, 'model_id');
  const modelArtifactSha256 = readString(value, 'model_artifact_sha256');
  const fileScope = readString(value, 'file_scope');
  const shell = readRecord(value, 'shell');
  const tools = readStringArray(value.tools, 32);
  const enforcedLayers = shell === null ? null : readStringArray(shell.enforced_layers, 16);
  const note = shell?.note;
  if (
    workspace === null ||
    modelId === null ||
    modelArtifactSha256 === null ||
    fileScope === null ||
    tools === null ||
    shell === null ||
    typeof shell.enabled !== 'boolean' ||
    typeof shell.mode !== 'string' ||
    enforcedLayers === null ||
    (note !== null && typeof note !== 'string') ||
    typeof value.camelid_network_tools !== 'boolean'
  ) return null;
  return {
    workspace,
    modelId,
    modelArtifactSha256,
    tools,
    fileScope,
    shell: {
      enabled: shell.enabled,
      mode: shell.mode,
      enforcedLayers,
      note: note as string | null,
    },
    camelidNetworkTools: value.camelid_network_tools,
  };
}

function readStringArray(value: unknown, maximum: number): readonly string[] | null {
  return Array.isArray(value)
    && value.length <= maximum
    && value.every((entry) => typeof entry === 'string')
    ? value as string[]
    : null;
}

function readPlan(value: unknown): readonly PlanProjection[] | null {
  if (!Array.isArray(value) || value.length > 20) return null;
  const steps: PlanProjection[] = [];
  for (const candidate of value) {
    if (candidate === null || typeof candidate !== 'object' || Array.isArray(candidate)) return null;
    const record = candidate as Readonly<Record<string, unknown>>;
    const status = record.status;
    const text = record.text;
    if (
      !['pending', 'in_progress', 'done'].includes(status as string) ||
      typeof text !== 'string' ||
      text.length > 160
    ) return null;
    steps.push({ status: status as PlanProjection['status'], text });
  }
  return steps;
}

function approvalStatus(event: RemoteEvent): ApprovalStatus | null {
  if (event.type === 'approval.expired') return 'expired';
  const decision = readString(event.payload, 'decision');
  switch (decision) {
    case 'allow_once':
      return 'allowed_once';
    case 'deny':
      return 'denied';
    case 'abort_turn':
      return 'aborted';
    case 'expired':
      return 'expired';
    case 'invalidated_by_cancel':
      return 'aborted';
    default:
      return null;
  }
}

function settlePendingApprovals(
  approvals: Readonly<Record<string, ApprovalProjection>>,
  status: ApprovalStatus,
): Readonly<Record<string, ApprovalProjection>> {
  return Object.fromEntries(Object.entries(approvals).map(([approvalId, approval]) => [
    approvalId,
    approval.status === 'pending' ? { ...approval, status } : approval,
  ]));
}

function readSessionStatus(value: Readonly<Record<string, unknown>>, key: string): SessionStatus | null {
  const status = readString(value, key);
  return status !== null && isSessionStatus(status) ? status : null;
}

function isSessionStatus(value: string): value is SessionStatus {
  return ['offline', 'idle', 'running', 'waiting_approval', 'cancelling', 'completed', 'aborted', 'failed'].includes(value);
}

function isTerminal(status: SessionStatus): boolean {
  return status === 'completed' || status === 'aborted' || status === 'failed';
}

function readString(value: Readonly<Record<string, unknown>>, key: string): string | null {
  const candidate = value[key];
  return typeof candidate === 'string' ? candidate : null;
}

function readRecord(
  value: Readonly<Record<string, unknown>>,
  key: string,
): Readonly<Record<string, unknown>> | null {
  const candidate = value[key];
  return candidate !== null && typeof candidate === 'object' && !Array.isArray(candidate)
    ? (candidate as Readonly<Record<string, unknown>>)
    : null;
}
