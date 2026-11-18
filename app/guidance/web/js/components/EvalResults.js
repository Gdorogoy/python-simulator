// What the Angle and Acro tabs show after a run. One layout for both drone styles: the server fills in what its env knows
// (angle: roll/pitch crash, chasers, stalled steps; acro: tilt crash, hover, speed, tilt) and a cell it cannot fill shows "-".
import { html } from '../html.js';
import { fmt } from '../format.js';
import { Collapsible, DataTable, Kpi, KeyValues, SummaryActions, TextBlock, Verdict } from './ui.js';
import { Player } from './Player.js';

const num = (value, digits) => (value == null ? '-' : fmt(value, digits));

// One flight: verdict, the 3D player, details and the timestamped log.
export function FlightResult({ result, distance, switchDist }) {
  const { data, track, meta, log, diag } = result;
  const extra = diag && diag.max_tilt_deg != null ? `, max tilt ${fmt(diag.max_tilt_deg, 0)} deg` : '';
  return html`
    <div className="results card">
      <div className="result-top">
        <${Verdict} reason=${data.reason === 'hit' ? 'Hit' : data.reason} />
        <span className="note">${data.steps} steps, final distance ${fmt(data.final_dist)} m, closest ${num(diag && diag.min_dist)} m${extra}</span>
      </div>
      <${Player} track=${track} meta=${meta} />
      <${Collapsible} title="Details">
        <${KeyValues} data=${{
          checkpoint: data.checkpoint, distance: data.distance ?? distance, switch_dist: data.switch_dist ?? switchDist,
          hit_radius: data.hit_radius, position_mode: data.position_mode, action_repeat: data.action_repeat ?? '-',
          recentre_z: data.recentre_z ?? 'off', chasers: data.threats ?? 'none',
          ...(data.runner ? { runner_speed: fmt(data.runner.speed, 1) + ' m/s', runner_escape: data.runner.escape_dist + ' m' } : {}),
          ...(diag || {}),
        }} />
      <//>
      <${Collapsible} title="Timestamped log"><${TextBlock} text=${(log || []).join('\n')} /><//>
    </div>`;
}

const BATCH_COLUMNS = [
  { head: '#', cell: (s, i) => i + 1 },
  { head: 'result', cell: (s) => s.reason },
  { head: 'start dist (m)', cell: (s) => num(s.dist) },
  { head: 'final dist (m)', cell: (s) => num(s.final_dist) },
  { head: 'closest (m)', cell: (s) => num(s.min_dist) },
  { head: 'handoff speed (m/s)', cell: (s) => num(s.handoff_speed ?? s.spawn_speed, 1) },
  { head: 'time (s)', cell: (s) => num(s.time_s, 1) },
  { head: 'mean speed (m/s)', cell: (s) => num(s.mean_speed, 1) },
  { head: 'max tilt (deg)', cell: (s) => num(s.tilt_max_deg, 0) },
  { head: 'nearest chaser (m)', cell: (s) => num(s.min_threat_dist) },
  { head: 'runner speed (m/s)', cell: (s) => num(s.runner_speed, 1) },
  { head: 'stalled steps', cell: (s) => s.stall_steps ?? '-' },
];

// Many flights: outcome counts, a copyable summary and a per-flight table.
export function BatchResult({ result, distance, switchDist }) {
  const { data } = result;
  const scenarios = data.scenarios || [];
  const totals = data.totals || {};
  const count = scenarios.length || Object.values(totals).reduce((a, b) => a + b, 0);
  const parked = scenarios.filter((s) => s.reason === 'timeout' && (s.stall_steps || 0) > 1000).length;
  const crash = (totals.attitude || 0) + (totals.tilt || 0);        // roll/pitch crash (angle) or tilt crash (acro)
  const rate = data.action_repeat ? ` | ${(60 * 4) / data.action_repeat} Hz policy` : '';
  const hitRule = data.hit_radius != null
    ? ` | hit ${fmt(data.hit_radius)} m` + (data.hit_rule === 'acro'
        ? ' first step / swept'
        : data.zone_time_s ? ` held ${fmt(data.zone_time_s, 2)} s` : '')
    : '';
  return html`
    <div className="results card">
      <div className="note">${data.checkpoint} | ${count} scenarios | distance ${distance} m, switch ${switchDist} m${hitRule}${rate}</div>
      <div className="kpis">
        <${Kpi} label="Hit" count=${totals.hit || 0} total=${count} tone="good" />
        <${Kpi} label="Attitude crash (roll / pitch / tilt)" count=${crash} total=${count} tone="bad" />
        ${(totals.caught || 0) > 0 && html`<${Kpi} label="Intercepted by chaser" count=${totals.caught} total=${count} tone="bad" />`}
        <${Kpi} label="Out of bounds" count=${totals.oob || 0} total=${count} tone="bad" />
        <${Kpi} label="Hover (stalled)" count=${totals.hover || 0} total=${count} tone="warn" />
        ${(totals.near_miss || 0) > 0 && html`<${Kpi} label="Near miss (circled too long)" count=${totals.near_miss} total=${count} tone="warn" />`}
        ${(totals.escape || 0) > 0 && html`<${Kpi} label="Escaped (runner got away)" count=${totals.escape} total=${count} tone="bad" />`}
        <${Kpi} label="Timeout" count=${totals.timeout || 0} total=${count} tone="warn" />
        ${parked > 0 && html`<${Kpi} label="of which parked < 1 m" count=${parked} total=${count} tone="warn" />`}
      </div>
      <${SummaryActions} text=${data.summary_text} url=${data.summary_url} />
      <${Collapsible} title="Per-scenario results"><${DataTable} columns=${BATCH_COLUMNS} rows=${scenarios} /><//>
      <${Collapsible} title="Summary text"><${TextBlock} text=${data.summary_text} /><//>
      <${Collapsible} title="Timestamped log"><${TextBlock} text=${(data.log || []).join('\n')} /><//>
    </div>`;
}
