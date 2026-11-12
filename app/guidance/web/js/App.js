// The page: title, two tabs, and the tab bodies. Both tabs stay mounted (just hidden) so switching
// tabs does not throw away a result or a running 3D replay.
import { html } from './html.js';
import { useCheckpointList, useStoredState } from './hooks.js';
import { AngleTab } from './tabs/AngleTab.js';
import { AcroTab } from './tabs/AcroTab.js';

const TABS = [
  { id: 'two', label: 'Angle eval' },
  { id: 'acro', label: 'Acro eval' },
];

export function App() {
  const [activeTab, setActiveTab] = useStoredState('tab', 'two');
  const checkpoints = useCheckpointList();

  return html`
    <div className="wrap">
      <header>
        <h1>Drone Policy Viewer</h1>
        <p>Fly checkpoints in the numpy simulator and replay the flight in 3D: orbit, zoom, scrub, pause, follow the drone (local tool, no auth, runs are synchronous).</p>
      </header>

      <nav className="tabs" role="tablist" aria-label="Mode">
        ${TABS.map((tab) => html`
          <button key=${tab.id} role="tab" aria-selected=${activeTab === tab.id} onClick=${() => setActiveTab(tab.id)}>${tab.label}</button>`)}
      </nav>

      <section className="panel" role="tabpanel" hidden=${activeTab !== 'two'}>
        <${AngleTab} checkpoints=${checkpoints} />
      </section>
      <section className="panel" role="tabpanel" hidden=${activeTab !== 'acro'}>
        <${AcroTab} checkpoints=${checkpoints} />
      </section>
    </div>`;
}
