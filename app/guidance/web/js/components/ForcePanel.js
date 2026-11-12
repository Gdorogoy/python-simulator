// The side panel of the 3D player: forces on the drone, what each rotor makes and what the commanded action asks of it,
// how close the commands are to their limits, and the angles between the drone and the target.
// `dyn` is one frame of numbers from PlayerScene.drawDynamics (null before the first frame); `meta` is track.dyn_meta.
import { html } from '../html.js';
import { fmt } from '../format.js';
import { ROTOR_COLORS } from '../player/PlayerScene.js';

const GRAVITY = 9.81;
const AT_LIMIT = 0.98;                                   // a command this close to its bound counts as "at the limit"
const swatch = (color) => '#' + color.toString(16).padStart(6, '0');
const magnitude = (v) => Math.hypot(v[0], v[1], v[2]);
const signed = (x, digits = 2) => (x == null ? '-' : (x >= 0 ? '+' : '') + Number(x).toFixed(digits));

function Row({ label, children, tone, title }) {
  return html`<tr className=${tone || ''} title=${title}><th>${label}</th><td>${children}</td></tr>`;
}

// One command against its bound: "+5.9 N (40%)", flagged when it sits on the bound.
function limitCell(value, low, high, unit, digits = 2) {
  const pct = value >= 0 ? value / high : value / low;          // share of the bound on that side
  const atLimit = Math.abs(pct) >= AT_LIMIT;
  return html`<span>${signed(value, digits)} ${unit} <span className=${atLimit ? 'pv-flag' : 'pv-dim'}>${atLimit ? 'AT LIMIT' : Math.round(Math.abs(pct) * 100) + '%'}</span></span>`;
}

export function ForcePanel({ dyn, meta }) {
  if (!dyn) return html`<aside className="pv-side"><p className="pv-dim">Press play or scrub the timeline.</p></aside>`;
  const acro = meta.style === 'acro';
  const { angles } = dyn;
  const cmdUnit = acro ? 'rad/s' : 'N·m';
  const axisNames = ['roll', 'pitch', 'yaw'];
  const hoverF = meta.rotor_hover_f;
  const netAccel = magnitude(dyn.accel);

  return html`
    <aside className="pv-side">
      <h4>Forces on the drone</h4>
      <table className="pv-tab"><tbody>
        <${Row} label="thrust (made / wanted)" title="Total of the four rotors, along the drone's up axis">${fmt(dyn.thrust, 1)} / ${fmt(dyn.thrustWant, 1)} N <span className="pv-dim">${fmt(dyn.thrust / meta.weight, 2)} x weight</span><//>
        <${Row} label="weight">${fmt(meta.weight, 1)} N<//>
        <${Row} label="drag">${fmt(magnitude(dyn.drag), 2)} N<//>
        <${Row} label="net force" title="thrust + weight + drag">${fmt(magnitude(dyn.net), 1)} N <span className="pv-dim">(${signed(dyn.net[0], 1)}, ${signed(dyn.net[1], 1)}, ${signed(dyn.net[2], 1)})</span><//>
        <${Row} label="acceleration">${fmt(netAccel, 1)} m/s² <span className="pv-dim">${fmt(netAccel / GRAVITY, 2)} g</span><//>
      </tbody></table>

      <h4>Rotors <span className="pv-dim">made / wanted</span></h4>
      <table className="pv-tab pv-rotors">
        <thead><tr><th></th><th>speed</th><th>thrust N</th><th>wanted N</th><th>of hover</th></tr></thead>
        <tbody>
          ${dyn.rotorF.map((made, k) => {
            const wanted = dyn.rotorWant[k];
            const saturated = wanted < 0;
            return html`<tr key=${k} className=${saturated ? 'bad' : ''}
              title=${saturated ? 'The commands ask this rotor to push down. It cannot: it just stops (0 N).' : ''}>
              <th><span className="pv-chip" style=${{ background: swatch(ROTOR_COLORS[k]) }}></span>R${k + 1}</th>
              <td>${fmt(dyn.rotorW[k], 0)}</td><td>${fmt(made, 2)}</td><td>${fmt(wanted, 2)}${saturated ? ' ✕' : ''}</td>
              <td>${Math.round((100 * made) / hoverF)}%</td></tr>`;
          })}
        </tbody>
      </table>

      <h4>${acro ? 'Commands (thrust, body rates)' : 'Commands (thrust, torques)'}</h4>
      <table className="pv-tab"><tbody>
        <${Row} label="thrust above hover" title="Action 0: thrust minus the hover thrust, clipped to the action bounds">${limitCell(dyn.action[0], meta.action_low[0], meta.action_high[0], 'N', 1)}<//>
        ${axisNames.map((name, a) => html`
          <${Row} key=${name} label=${(acro ? name + ' rate' : name + ' torque')}>${limitCell(dyn.action[a + 1], meta.action_low[a + 1], meta.action_high[a + 1], cmdUnit, 2)}<//>`)}
        ${axisNames.map((name, a) => html`
          <${Row} key=${'t' + name} label=${name + ' torque (made / wanted)'}>${signed(dyn.torque[a], 3)} / ${signed(dyn.torqueWant[a], 3)} N·m<//>`)}
      </tbody></table>

      <h4>Angles to the target</h4>
      <table className="pv-tab"><tbody>
        <${Row} label="thrust axis ↔ target" tone="arc-thrust" title="Angle between where the rotors push and the line to the target (yellow-green arc)">${fmt(angles.thrust_los, 1)}°<//>
        <${Row} label="velocity ↔ target" tone="arc-vel" title="Angle between where the drone is moving and the line to the target (blue arc)">${angles.vel_los == null ? '-' : fmt(angles.vel_los, 1) + '°'}<//>
        <${Row} label="tilt from vertical">${fmt(angles.tilt, 1)}°<//>
        <${Row} label="yaw error">${signed(angles.yaw_err, 1)}°<//>
        <${Row} label="target bearing / elevation" title="Where the target is, seen from the drone's own frame (x forward, z up)">${signed(angles.az, 0)}° / ${signed(angles.el, 0)}°<//>
      </tbody></table>
    </aside>`;
}
