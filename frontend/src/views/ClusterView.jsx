import { useCallback, useRef, useState } from 'react'
import { useClusterTopology } from '../hooks/useClusterTopology'
import { TopologyCanvas } from '../components/cluster/TopologyCanvas'
import { NodeInventory } from '../components/cluster/NodeInventory'
import { NodeInspector } from '../components/cluster/NodeInspector'
import { ClusterDrawer } from '../components/cluster/ClusterDrawer'
import { AddServerWizard } from '../components/cluster/AddServerWizard'
import { DiscoverDevices } from '../components/cluster/DiscoverDevices'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import {
  IconPlus, IconNetwork, IconCheckCircle, IconDownload, IconGrid, IconChart, IconTrash,
} from '../components/ui/icons'

export default function ClusterView({ showNotice }) {
  // Validation results open in the drawer's Validation tab below, so the
  // hook's "see the events drawer" toast would point at the wrong place.
  const notify = useCallback((message, tone) => {
    if (typeof message === 'string' && message.startsWith('Cluster validation complete')) return
    showNotice?.(message, tone)
  }, [showNotice])
  const cluster = useClusterTopology({ showNotice: notify })
  const canvasRef = useRef(null)
  const [wizardOpen, setWizardOpen] = useState(false)
  const [discoverOpen, setDiscoverOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerTab, setDrawerTab] = useState('events')
  const [clearOpen, setClearOpen] = useState(false)

  const resetLayout = () => { cluster.applyAutoLayout(); window.setTimeout(() => canvasRef.current?.fit(), 60) }

  const openDrawer = (tab) => { setDrawerTab(tab); setDrawerOpen(true) }
  const runValidation = async () => {
    await cluster.validateCluster()
    openDrawer('validation')
  }

  const inspectorActions = {
    testNode: cluster.testNode,
    startWorker: cluster.startWorker,
    stopWorker: cluster.stopWorker,
    restartWorker: cluster.restartWorker,
    removeNode: cluster.removeNode,
    updateNode: cluster.updateNode,
    testConnection: cluster.testConnection,
    updateConnection: cluster.updateConnection,
    removeConnection: cluster.removeConnection,
  }

  return (
    <div className="cluster-view">
      <header className="cluster-header">
        <div className="cluster-header__copy">
          <p className="cxv-kicker"><IconNetwork size={14} /> Cluster</p>
          <h1>Cluster</h1>
          <p className="cxv-sub">Connect Macs, Windows PCs, Linux servers, and Raspberry Pis into one local Camelid compute fabric.</p>
        </div>
        <div className="cluster-header__actions">
          <div className="cluster-header__primary">
            <Button variant="primary" icon={<IconPlus size={16} />} onClick={() => setWizardOpen(true)}>Add server</Button>
            <Button variant="tonal" icon={<IconNetwork size={16} />} onClick={() => setDiscoverOpen(true)}>Discover devices</Button>
            <Button variant="tonal" icon={<IconCheckCircle size={16} />} onClick={runValidation}>Validate cluster</Button>
          </div>
          <div className="cluster-header__secondary">
            <span className="cluster-header__autosave">Changes save automatically</span>
            <Button variant="ghost" size="sm" icon={<IconDownload size={15} />} onClick={cluster.exportTopology}>Export diagram</Button>
            <Button variant="ghost" size="sm" icon={<IconGrid size={15} />} onClick={resetLayout}>Reset layout</Button>
            <Button variant="ghost" size="sm" icon={<IconChart size={15} />} onClick={() => openDrawer('events')}>View logs</Button>
            {cluster.nodes.length > 0 && (
              <Button variant="ghost" size="sm" icon={<IconTrash size={15} />} onClick={() => setClearOpen(true)}>Clear cluster</Button>
            )}
          </div>
        </div>
      </header>

      <div className="cluster-body">
        <NodeInventory
          nodes={cluster.nodes}
          summary={cluster.summary}
          selection={cluster.selection}
          onSelect={cluster.select}
          onAddServer={() => setWizardOpen(true)}
        />

        <TopologyCanvas
          ref={canvasRef}
          nodes={cluster.nodes}
          connections={cluster.connections}
          selection={cluster.selection}
          busyIds={cluster.busyIds}
          onSelect={cluster.select}
          onMoveNode={cluster.moveNode}
          onAddConnection={cluster.addConnection}
          onAutoLayout={resetLayout}
          onAddServer={() => setWizardOpen(true)}
          onLoadSample={cluster.loadSample}
        />

        <NodeInspector
          selectedNode={cluster.selectedNode}
          selectedConnection={cluster.selectedConnection}
          nodes={cluster.nodes}
          actions={inspectorActions}
          onViewLogs={() => openDrawer('events')}
        />
      </div>

      <ClusterDrawer
        events={cluster.events}
        issues={cluster.issues}
        summary={cluster.summary}
        open={drawerOpen}
        tab={drawerTab}
        onSelectTab={setDrawerTab}
        onToggle={() => setDrawerOpen((v) => !v)}
      />

      <AddServerWizard open={wizardOpen} onClose={() => setWizardOpen(false)} onAdd={(node) => { cluster.addNode(node); setWizardOpen(false) }} />
      <DiscoverDevices open={discoverOpen} onClose={() => setDiscoverOpen(false)} onDiscover={cluster.discover} onAdd={(partial) => cluster.addNode(partial)} />

      <ConfirmDialog
        open={clearOpen}
        title="Clear the cluster topology?"
        detail={`This removes all ${cluster.nodes.length} node${cluster.nodes.length === 1 ? '' : 's'} and ${cluster.connections.length} link${cluster.connections.length === 1 ? '' : 's'} from this diagram, including the sample fabric. It only clears what is drawn here — no machine is touched and nothing is uninstalled. Export the diagram first if you want a copy.`}
        confirmLabel="Clear topology"
        onCancel={() => setClearOpen(false)}
        onConfirm={() => { cluster.resetTopology(); setClearOpen(false); showNotice?.('Cluster topology cleared.', 'info') }}
      />
    </div>
  )
}
