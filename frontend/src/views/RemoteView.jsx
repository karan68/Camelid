import { useEffect, useState } from 'react'
import QRCode from 'qrcode'
import { Button } from '../components/ui/Button'
import { Card, CardBody, CardHeader } from '../components/ui/Card'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { EmptyState } from '../components/ui/EmptyState'
import { StatusDot } from '../components/ui/StatusDot'
import { IconNetwork, IconRefresh, IconStop, IconWarning } from '../components/ui/icons'

const API_PATH = '/api/agent/remote'

function formatTime(value) {
  if (!value) return 'Never'
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}

function stateTone(state) {
  if (state === 'running') return 'ready'
  if (state === 'waiting_approval' || state === 'cancelling') return 'warn'
  return 'neutral'
}

export default function RemoteView({ showNotice }) {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [pendingRevoke, setPendingRevoke] = useState(null)
  const [confirmDisable, setConfirmDisable] = useState(false)
  const [qrDataUrl, setQrDataUrl] = useState('')
  const [qrExpiresAt, setQrExpiresAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())

  const refresh = async ({ silent = false } = {}) => {
    if (!silent) setLoading(true)
    try {
      const response = await fetch(`${API_PATH}/status`, { credentials: 'same-origin' })
      if (response.status === 404) {
        setStatus(null)
        setError('Start Camelid with agent host to manage remote control here.')
        return
      }
      if (!response.ok) throw new Error('status unavailable')
      setStatus(await response.json())
      setError('')
    } catch {
      setError('Remote host status is unavailable.')
    } finally {
      if (!silent) setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const timer = window.setInterval(() => refresh({ silent: true }), 3000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (status?.pairing?.state !== 'offered') {
      setQrDataUrl('')
      setQrExpiresAt(0)
    }
  }, [status?.pairing?.state])

  const invoke = async (path, success, body = null) => {
    setBusy(true)
    try {
      const response = await fetch(`${API_PATH}${path}`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: body ? { 'content-type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      })
      if (!response.ok) throw new Error('remote control action failed')
      showNotice?.(success, 'success')
      await refresh()
    } catch {
      showNotice?.('The running remote host could not complete that action.', 'error')
    } finally {
      setBusy(false)
    }
  }

  const session = status?.session
  const pairing = status?.pairing
  const capability = session?.capability_snapshot || {}
  const shell = capability.shell || {}
  const tools = Array.isArray(capability.tools) ? capability.tools : []
  const authorizedDevices = status?.devices?.filter((device) => device.status === 'authorized') || []
  const pairingSeconds = Math.max(0, Math.ceil(((qrExpiresAt || pairing?.expires_at_unix_ms || 0) - now) / 1000))

  const createPairing = async () => {
    setBusy(true)
    try {
      const response = await fetch(`${API_PATH}/pairing`, { method: 'POST', credentials: 'same-origin' })
      if (!response.ok) throw new Error('pairing offer failed')
      const offer = await response.json()
      const dataUrl = await QRCode.toDataURL(offer.qr_payload, {
        errorCorrectionLevel: 'M',
        margin: 2,
        width: 280,
        color: { dark: '#111827', light: '#ffffff' },
      })
      setQrDataUrl(dataUrl)
      setQrExpiresAt(offer.expires_at_unix_ms)
      setNow(Date.now())
      await refresh({ silent: true })
    } catch {
      setQrDataUrl('')
      setQrExpiresAt(0)
      showNotice?.('Could not create a pairing offer. Cancel the current offer or wait for expiry.', 'error')
    } finally {
      setBusy(false)
    }
  }

  const cancelPairing = async () => {
    setQrDataUrl('')
    setQrExpiresAt(0)
    await invoke('/pairing/cancel', 'Pairing cancelled.')
  }

  const decidePairing = async (accepted) => {
    if (!pairing?.confirmation_id) return
    await invoke(
      `/pairing/${pairing.confirmation_id}/confirm`,
      accepted ? `Paired ${pairing.device_label}.` : `Rejected ${pairing.device_label}.`,
      { accepted },
    )
  }

  return (
    <section className="remote-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconNetwork size={14} /> Remote</p>
          <h1>Remote Control</h1>
          <p className="cxv-sub">Local authority for the explicitly armed host.</p>
        </div>
        <div className="cxv-head__actions">
          <Button variant="ghost" size="sm" icon={<IconRefresh size={16} />} onClick={refresh} loading={loading}>Refresh</Button>
          <StatusDot tone={status?.configured ? 'ready' : 'neutral'} label={status?.configured ? 'Host armed' : 'Not armed'} />
        </div>
      </header>

      {error && <EmptyState icon={<IconNetwork size={22} />} title="Remote host unavailable" description={error} />}

      {status?.configured && (
        <>
          <div className="cxv-stat-grid remote-view__stats">
            <div className="cxv-stat"><span>Host</span><strong>Armed</strong><small>{status.host_id}</small></div>
            <div className="cxv-stat"><span>Session</span><strong>{session?.state || 'Unknown'}</strong><small>{session ? `Event ${session.last_event_sequence}` : 'No session'}</small></div>
            <div className="cxv-stat"><span>Paired devices</span><strong>{authorizedDevices.length}</strong><small>{status.devices.length} registered</small></div>
            <div className="cxv-stat"><span>Relay</span><strong>{status.relay_url ? 'Configured' : 'Unavailable'}</strong><small>{status.relay_url || 'No route'}</small></div>
          </div>

          <div className="remote-view__grid">
            <Card>
              <CardHeader icon={<IconNetwork size={20} />} eyebrow="Session" title="Current authority" actions={<StatusDot tone={stateTone(session?.state)} label={session?.state || 'Unknown'} />} />
              <CardBody>
                <dl className="remote-view__details">
                  <div><dt>Host identity</dt><dd>{status.host_id}</dd></div>
                  <div><dt>Session ID</dt><dd>{session?.session_id || 'No active session'}</dd></div>
                  <div><dt>Last update</dt><dd>{formatTime(session?.updated_at_unix_ms)}</dd></div>
                </dl>
              </CardBody>
            </Card>
            <Card tone="muted">
              <CardHeader icon={<IconWarning size={20} />} eyebrow="Emergency" title="Disable remote control" />
              <CardBody>
                <p className="remote-view__copy">Revokes every paired device, closes live device connections, cancels the active remote turn, and invalidates pending approvals.</p>
                <Button variant="danger" icon={<IconStop size={16} />} onClick={() => setConfirmDisable(true)} disabled={busy || authorizedDevices.length === 0}>Emergency disable</Button>
              </CardBody>
            </Card>
          </div>

          <Card className="remote-view__pairing">
            <CardHeader
              eyebrow="Pairing"
              title="Add a mobile device"
              actions={<StatusDot tone={pairing?.state === 'awaiting_confirmation' ? 'warn' : pairing?.state === 'offered' ? 'ready' : 'neutral'} label={pairing?.state === 'awaiting_confirmation' ? 'Confirmation required' : pairing?.state === 'offered' ? 'Offer active' : 'Idle'} />}
            />
            <CardBody>
              {!pairing && (
                <div className="remote-view__pairing-idle">
                  <p className="remote-view__copy">Create one five-minute QR offer. A device receives no authority until its encrypted identity and fingerprint are confirmed here.</p>
                  <Button variant="primary" icon={<IconNetwork size={16} />} onClick={createPairing} loading={busy} disabled={busy}>Pair new device</Button>
                </div>
              )}

              {pairing?.state === 'offered' && (
                <div className="remote-view__offer">
                  {qrDataUrl ? (
                    <img className="remote-view__qr" src={qrDataUrl} alt="Pairing QR code for Camelid Mobile" width="280" height="280" />
                  ) : (
                    <div className="remote-view__qr-missing" role="status">This offer was created in another UI session. Cancel it and create a new QR here.</div>
                  )}
                  <div className="remote-view__offer-copy">
                    <strong>Scan with Camelid Mobile</strong>
                    <p>The offer expires in <span aria-live="polite">{pairingSeconds}s</span>. Keep this screen local and cancel the offer if the QR may have been photographed.</p>
                    <Button variant="outline" onClick={cancelPairing} loading={busy} disabled={busy}>Cancel offer</Button>
                  </div>
                </div>
              )}

              {pairing?.state === 'awaiting_confirmation' && (
                <div className="remote-view__confirmation" role="status" aria-live="polite">
                  <div>
                    <span className="remote-view__confirmation-label">Device requesting access</span>
                    <strong>{pairing.device_label}</strong>
                  </div>
                  <div>
                    <span className="remote-view__confirmation-label">Authentication fingerprint</span>
                    <code>{pairing.authentication_fingerprint}</code>
                  </div>
                  <p>Confirm only if this label and fingerprint match the phone in your hand. Approval grants this device the capability snapshot shown below.</p>
                  <div className="remote-view__confirmation-actions">
                    <Button variant="primary" onClick={() => decidePairing(true)} loading={busy} disabled={busy}>Approve device</Button>
                    <Button variant="danger" onClick={() => decidePairing(false)} disabled={busy}>Reject</Button>
                  </div>
                </div>
              )}
            </CardBody>
          </Card>

          <Card className="remote-view__capabilities">
            <CardHeader eyebrow="Capability snapshot" title="Authority fixed when armed" actions={<span className="cxv-section__count">{capability.profile || 'unavailable'}</span>} />
            <CardBody>
              <p className="remote-view__copy">These values are the host&apos;s durable session snapshot. Changing the local UI does not widen remote authority.</p>
              <dl className="remote-view__capability-grid">
                <div><dt>Workspace scope</dt><dd>{capability.workspace || 'Unavailable'}</dd></div>
                <div><dt>File scope</dt><dd>{capability.file_scope || 'Unavailable'}</dd></div>
                <div><dt>Network tools</dt><dd>{capability.camelid_network_tools ? 'Enabled' : 'Disabled'}</dd></div>
                <div><dt>Shell enforcement</dt><dd>{shell.enabled ? `${shell.mode || 'enabled'}${Array.isArray(shell.enforced_layers) && shell.enforced_layers.length ? ` (${shell.enforced_layers.join(', ')})` : ''}` : 'Disabled'}</dd></div>
              </dl>
              {shell.note && <p className="remote-view__shell-note">{shell.note}</p>}
              <div className="remote-view__tools"><span>Enabled tools</span>{tools.length ? tools.map((tool) => <code key={tool}>{tool}</code>) : <strong>None</strong>}</div>
            </CardBody>
          </Card>

          <Card className="remote-view__devices">
            <CardHeader eyebrow="Devices" title="Paired devices" actions={<span className="cxv-section__count">{status.devices.length} registered</span>} />
            <CardBody>
              {status.devices.length === 0 ? <p className="remote-view__copy">No device has completed local confirmation.</p> : (
                <div className="remote-view__table" role="table" aria-label="Paired remote devices">
                  {status.devices.map((device) => (
                    <div className="remote-view__row" role="row" key={device.device_id}>
                      <div role="cell"><strong>{device.label}</strong><small>{device.device_id}</small></div>
                      <div role="cell"><StatusDot tone={device.status === 'authorized' ? 'ready' : 'neutral'} label={device.status} /></div>
                      <div role="cell"><span>Last seen</span><strong>{formatTime(device.last_seen_at_unix_ms)}</strong></div>
                      <div role="cell">{device.status === 'authorized' && <Button variant="outline" size="sm" onClick={() => setPendingRevoke(device)} disabled={busy}>Revoke</Button>}</div>
                    </div>
                  ))}
                </div>
              )}
            </CardBody>
          </Card>
        </>
      )}

      <ConfirmDialog open={Boolean(pendingRevoke)} title={`Revoke ${pendingRevoke?.label || 'device'}?`} detail="This immediately disconnects that device. If it owns the active remote turn, Camelid cancels the turn and invalidates pending approvals." confirmLabel="Revoke device" busy={busy} onCancel={() => { if (!busy) setPendingRevoke(null) }} onConfirm={async () => { const device = pendingRevoke; setPendingRevoke(null); await invoke(`/devices/${device.device_id}/revoke`, `Revoked ${device.label}.`) }} />
      <ConfirmDialog open={confirmDisable} title="Emergency disable remote control?" detail="This revokes all paired devices, disconnects them, cancels remote work, and invalidates pending approvals. Local files and completed changes are not rolled back." confirmLabel="Emergency disable" busy={busy} onCancel={() => { if (!busy) setConfirmDisable(false) }} onConfirm={async () => { setConfirmDisable(false); await invoke('/disable', 'Remote control disabled and paired devices revoked.') }} />
    </section>
  )
}