export type ApprovalGateResult =
  | { authorized: true }
  | { authorized: false; reason: 'not_configured' | 'unavailable' | 'cancelled' | 'failed' };

export interface LocalAuthenticationPort {
  hasHardwareAsync(): Promise<boolean>;
  isEnrolledAsync(): Promise<boolean>;
  authenticateAsync(options: {
    promptMessage: string;
    promptDescription: string;
    biometricsSecurityLevel: 'strong';
    disableDeviceFallback: boolean;
    fallbackLabel: string;
    cancelLabel: string;
  }): Promise<{ success: true } | { success: false; error: string }>;
}

export async function authorizeApproval(
  required: boolean,
  authentication: LocalAuthenticationPort,
): Promise<ApprovalGateResult> {
  if (!required) return { authorized: true };
  const [hasHardware, isEnrolled] = await Promise.all([
    authentication.hasHardwareAsync(),
    authentication.isEnrolledAsync(),
  ]);
  if (!hasHardware || !isEnrolled) return { authorized: false, reason: 'unavailable' };

  const result = await authentication.authenticateAsync({
    promptMessage: 'Confirm remote approval',
    promptDescription: 'Authenticate to allow this action once.',
    biometricsSecurityLevel: 'strong',
    disableDeviceFallback: true,
    fallbackLabel: '',
    cancelLabel: 'Cancel',
  });
  if (result.success) return { authorized: true };
  if (['user_cancel', 'app_cancel', 'system_cancel'].includes(result.error)) {
    return { authorized: false, reason: 'cancelled' };
  }
  return { authorized: false, reason: 'failed' };
}
