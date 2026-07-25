import { authorizeApproval, LocalAuthenticationPort } from './approvalGate';

function authentication(
  overrides: Partial<LocalAuthenticationPort> = {},
): LocalAuthenticationPort {
  return {
    hasHardwareAsync: async () => true,
    isEnrolledAsync: async () => true,
    authenticateAsync: async () => ({ success: true }),
    ...overrides,
  };
}

describe('approval authentication gate', () => {
  test('does not invoke native authentication when the gate is disabled', async () => {
    const authenticateAsync = jest.fn(async () => ({ success: true as const }));
    await expect(authorizeApproval(false, authentication({ authenticateAsync }))).resolves.toEqual({
      authorized: true,
    });
    expect(authenticateAsync).not.toHaveBeenCalled();
  });

  test('fails closed when strong local authentication is unavailable', async () => {
    await expect(
      authorizeApproval(true, authentication({ isEnrolledAsync: async () => false })),
    ).resolves.toEqual({ authorized: false, reason: 'unavailable' });
  });

  test('requests strong biometrics without device fallback', async () => {
    const authenticateAsync = jest.fn(async () => ({ success: true as const }));
    await expect(authorizeApproval(true, authentication({ authenticateAsync }))).resolves.toEqual({
      authorized: true,
    });
    expect(authenticateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        biometricsSecurityLevel: 'strong',
        disableDeviceFallback: true,
        fallbackLabel: '',
      }),
    );
  });

  test.each([
    ['user_cancel', 'cancelled'],
    ['system_cancel', 'cancelled'],
    ['authentication_failed', 'failed'],
  ] as const)('maps %s without granting authority', async (error, reason) => {
    await expect(
      authorizeApproval(
        true,
        authentication({ authenticateAsync: async () => ({ success: false, error }) }),
      ),
    ).resolves.toEqual({ authorized: false, reason });
  });
});
