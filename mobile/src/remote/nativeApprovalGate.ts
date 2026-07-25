import * as LocalAuthentication from 'expo-local-authentication';

import { authorizeApproval } from './approvalGate';

export function authorizeNativeApproval(required: boolean) {
  return authorizeApproval(required, LocalAuthentication);
}
