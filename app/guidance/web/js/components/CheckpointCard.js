// The "Checkpoint" card: pick a .pt the server knows about, or upload one from your computer.
import { html } from '../html.js';
import { Card, Segmented } from './ui.js';

// `choice` comes from useCheckpointChoice() (see hooks.js). `note` is the grey line under the picker,
// `children` is any extra text (used by the Acro tab to explain how it scores).
export function CheckpointCard({ choice, note, children }) {
  const noFiles = choice.list.length === 0;
  return html`
    <${Card} title="Checkpoint">
      <${Segmented} label="Checkpoint source" value=${choice.source} onChange=${choice.setSource}
        options=${[{ value: 'pick', label: 'Pick from server' }, { value: 'upload', label: 'Upload .pt' }]} />
      ${choice.source === 'pick' && html`
        <div>
          <select aria-label="Checkpoint on the server" value=${choice.selected} onChange=${(e) => choice.setSelected(e.target.value)}>
            ${noFiles && html`<option value="">(no .pt files found under runs/ or app/control/)</option>`}
            ${choice.list.map((path) => html`<option key=${path} value=${path}>${path}</option>`)}
          </select>
        </div>`}
      ${choice.source === 'upload' && html`
        <div><input type="file" accept=".pt" aria-label="Checkpoint file" onChange=${(e) => choice.setFile(e.target.files[0] || null)} /></div>`}
      <p className="note">${note}</p>
      ${children}
    <//>`;
}
