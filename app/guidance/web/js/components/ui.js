// Small building blocks used by every tab: cards, form fields, status line, result widgets.
import { Fragment, useState } from 'react';
import { html } from '../html.js';
import { percent } from '../format.js';

export function Card({ title, children }) {
  return html`<div className="card">${title && html`<h2>${title}</h2>`}${children}</div>`;
}

// A labelled number box. `onChange` receives the new text (a string).
export function NumberField({ label, hint, value, onChange, disabled, ...inputProps }) {
  return html`
    <label className="field">${label}
      <input type="number" value=${value} disabled=${disabled} onChange=${(e) => onChange(e.target.value)} ...${inputProps} />
      ${hint && html`<span className="hint">${hint}</span>`}
    </label>`;
}

// A labelled drop-down. `options` is a list of { value, label }.
export function SelectField({ label, hint, value, onChange, disabled, options }) {
  return html`
    <label className="field">${label}
      <select value=${value} disabled=${disabled} onChange=${(e) => onChange(e.target.value)}>
        ${options.map((o) => html`<option key=${o.value} value=${o.value}>${o.label}</option>`)}
      </select>
      ${hint && html`<span className="hint">${hint}</span>`}
    </label>`;
}

// A checkbox with a bold title and a grey explanation under it.
export function Toggle({ checked, onChange, disabled, title, children, style }) {
  return html`
    <div className="toggle" style=${style}>
      <label style=${{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
        <input type="checkbox" checked=${checked} disabled=${disabled} onChange=${(e) => onChange(e.target.checked)} style=${{ marginTop: '3px' }} />
        <span><b>${title}</b><span className="note">${children}</span></span>
      </label>
    </div>`;
}

// Two or more buttons where exactly one is "pressed". options = [{ value, label }].
export function Segmented({ options, value, onChange, label, className = '' }) {
  return html`
    <div className=${'seg ' + className} role="group" aria-label=${label}>
      ${options.map((o) => html`
        <button key=${o.value} type="button" aria-pressed=${value === o.value} onClick=${() => onChange(o.value)}>${o.label}</button>`)}
    </div>`;
}

// Quick-fill buttons. items = [{ key, label }], onPick gets the key.
export function Chips({ items, onPick }) {
  return html`
    <div className="chips" aria-label="Presets">
      ${items.map((item) => html`<button key=${item.key} className="chip" type="button" onClick=${() => onPick(item.key)}>${item.label}</button>`)}
    </div>`;
}

// The line under the buttons: spinner while busy, green when done, red on error.
export function Status({ status }) {
  return html`
    <div className=${'status ' + status.kind} role="status" aria-live="polite">
      ${status.kind === 'busy' && html`<span className="spin"></span>`}
      ${status.message && html`<span>${status.message}</span>`}
    </div>`;
}

// One big percentage card: "Hit 82% (62/75)".
export function Kpi({ label, count, total, tone }) {
  const value = percent(count, total);
  return html`
    <div className=${'kpi ' + (tone || '')}>
      <div className="v">${value.toFixed(0)}%</div>
      <div className="l">${label} (${count}/${total})</div>
      <div className="bar"><i style=${{ width: value + '%' }}></i></div>
    </div>`;
}

// Coloured pill with the episode outcome ("Hit" is green, "timeout" orange, anything else red - "Intercepted" = caught by a chasing drone).
export function Verdict({ reason }) {
  const tone = reason === 'Hit' ? 'good' : reason === 'timeout' ? 'warn' : 'bad';
  return html`<span className=${'verdict ' + tone}>${reason}</span>`;
}

export function Collapsible({ title, children }) {
  return html`<details><summary>${title}</summary>${children}</details>`;
}

export function TextBlock({ text }) {
  return html`<pre className="out">${text}</pre>`;
}

// A name/value list from an object. Keys whose value is undefined are skipped, objects are shown as JSON.
export function KeyValues({ data }) {
  const entries = Object.entries(data).filter(([, value]) => value !== undefined);
  return html`
    <dl className="kv">
      ${entries.map(([key, value]) => html`
        <${Fragment} key=${key}>
          <dt>${key}</dt>
          <dd>${typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value)}</dd>
        </${Fragment}>`)}
    </dl>`;
}

// "Copy summary" + "Open saved summary" under a batch result.
export function SummaryActions({ text, url }) {
  const [label, setLabel] = useState('Copy summary');
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setLabel('Copied');
      setTimeout(() => setLabel('Copy summary'), 1500);
    } catch (e) {
      setLabel('Copy failed');
    }
  }
  return html`
    <div className="rowbtn">
      <button className="linkbtn" type="button" onClick=${copy}>${label}</button>
      ${url && html`<a href=${url} target="_blank" rel="noopener">Open saved summary</a>`}
    </div>`;
}

// A scrollable table. columns = [{ head: 'closest (m)', cell: (row, index) => ... }].
export function DataTable({ columns, rows }) {
  return html`
    <div className="tablewrap">
      <table>
        <thead><tr>${columns.map((c) => html`<th key=${c.head}>${c.head}</th>`)}</tr></thead>
        <tbody>
          ${rows.map((row, index) => html`
            <tr key=${index}>${columns.map((c) => html`<td key=${c.head}>${c.cell(row, index)}</td>`)}</tr>`)}
        </tbody>
      </table>
    </div>`;
}
