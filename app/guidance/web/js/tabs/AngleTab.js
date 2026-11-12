// "Angle eval" tab: angle-style drones (the action is roll/pitch/yaw torque).
// Same page as the Acro tab (shared cards and result views); only the drone-specific fields differ.
// Flow: fill the form -> "Run & play in 3D" (one flight) or "Run test batch" (many flights, numbers only).
import { useEffect, useState } from 'react';
import { html } from '../html.js';
import { postForm } from '../api.js';
import { addHandoffFields, useCheckpointChoice, useHandoffForm, useRunner, useStoredState } from '../hooks.js';
import { Card, Chips, NumberField, Status, Toggle } from '../components/ui.js';
import { CheckpointCard } from '../components/CheckpointCard.js';
import { HandoffFields, HandoffToggle } from '../components/HandoffFields.js';
import { ChasersCard, THREAT_DEFAULTS, addChaserFields, useChasers } from '../components/ChasersCard.js';
import { BatchResult, FlightResult } from '../components/EvalResults.js';

// localStorage keys (kept the same as the old page so saved values still load)
const STORAGE_KEYS = {
  handoffOn: 'tpHandoffOn', distance: 'tpDistance', switchDist: 'tpSwitchDist', cruise: 'tpCruise',
  handoffSpeed: 'tpHandoffSpeed', lateral: 'tpLateral', tilt: 'tpTilt', runs: 'tpNRuns', seed: 'tpSeed',
};
const DEFAULTS = { distance: '250', switchDist: '40', cruise: '18', lateral: '0', tilt: '0' };
const PRESETS = {
  rtl: { label: 'Isaac RTL checkpoint (250 m, switch 40)', distance: 250, switchDist: 40 },
  short: { label: 'Short test (30 m, switch 20)', distance: 30, switchDist: 20 },
};

// One sentence about what the selected checkpoint was trained with.
function describeCheckpoint(info) {
  const parts = [];
  if (info.distance_high != null) parts.push(`trained for ${info.distance_low}-${info.distance_high} m`);
  if (info.max_speed != null) parts.push(`max speed ${info.max_speed} m/s`);
  if (info.spawn_dist_low != null) parts.push(`spawn leg ${info.spawn_dist_low}-${info.spawn_dist_high} m`);
  if (info.num_threats) parts.push(`${info.active_threats} of ${info.num_threats} chasing drone slots on (speed ${info.threat_speed ?? '?'} m/s)`);
  parts.push(`position input: ${info.obs_position_mode}`);
  let text = parts.join(', ');
  if (info.distance_high != null) text += ` | switch distance should be <= ${info.distance_high} m`;
  if (info.lateral_frac != null) text += ` | handoff sideways ${info.lateral_frac}, tilt ${info.handoff_tilt_deg} deg (pre-set)`;
  return text;
}

function describeRecentre(info) {
  if (info.obs_position_mode === 'height') return 'This checkpoint does not see x,y (height mode), so there is nothing to re-centre.';
  if (info.recentred) return 'Turned on: this checkpoint was trained with the handoff re-centred to x=y=0.';
  return 'Off: this run did not re-centre.';
}

function budgetHint(distance) {
  const meters = parseFloat(distance) || 0;
  if (meters > 5000) return 'Past 5000 m the step budget is clamped; the episode may time out before arriving.';
  if (meters > 500) return `At ${meters} m expect real minutes per run (more for a batch).`;
  return '';
}

export function AngleTab({ checkpoints }) {
  const choice = useCheckpointChoice('tp', checkpoints);
  const form = useHandoffForm(STORAGE_KEYS, DEFAULTS);
  const chasers = useChasers('tp');
  const [recentre, setRecentre] = useState(false);
  const [recentreZ, setRecentreZ] = useStoredState('tpRecentreZ', '20');
  const [hitRadius, setHitRadius] = useStoredState('tpHitRadius', '');
  const { status, busy, run, setStatus } = useRunner();
  const [result, setResult] = useState(null);       // { kind: 'flight' | 'batch', data, ... }

  // When the selected checkpoint's run config arrives, pre-fill the fields it knows about.
  const info = choice.info;
  useEffect(() => {
    if (!info) { setRecentre(false); return; }
    if (info.lateral_frac != null) {
      form.setLateral(String(info.lateral_frac));
      form.setTilt(String(info.handoff_tilt_deg ?? 0));
    }
    setRecentre(info.obs_position_mode !== 'height' && !!info.recentred);
    if (info.num_threats) {          // the run trained against chasing drones: start from the same setup
      chasers.setThreats(String(info.active_threats));
      if (info.threat_speed != null) chasers.setSpeed(String(info.threat_speed));
      if (info.threat_spread != null) chasers.setSpread(String(info.threat_spread));
      chasers.setSpeedMin(info.threat_speed_min != null ? String(info.threat_speed_min) : THREAT_DEFAULTS.speedMin);
      chasers.setSpawnMin(String(info.threat_spawn_min ?? THREAT_DEFAULTS.spawnMin));
      chasers.setSpawn(['base', 'geometry'].includes(info.threat_spawn) ? info.threat_spawn : THREAT_DEFAULTS.spawn);     // 'sphere' (old runs) = uniform
      const gw = (info.threat_geometry_weights || [1, 1]).slice(-2);     // old runs saved (parallel, head-on, tail): keep the last two
      chasers.setWHeadOn(String(gw[0])); chasers.setWTail(String(gw[1]));
      chasers.setReaction(String(info.threat_reaction_s ?? THREAT_DEFAULTS.reaction));
      const tg = info.threat_tail_gap || [15, 40];
      chasers.setTailLo(String(tg[0])); chasers.setTailHi(String(tg[1]));
      chasers.setJitter(String(info.threat_jitter ?? THREAT_DEFAULTS.jitter));
      chasers.setBuffer(String(info.threat_buffer ?? THREAT_DEFAULTS.buffer));
      chasers.setRadius(info.threat_radius != null ? String(info.threat_radius) : THREAT_DEFAULTS.radius);
      const after = info.threat_release_after || [Number(THREAT_DEFAULTS.releaseLo), Number(THREAT_DEFAULTS.releaseHi)];
      chasers.setReleaseLo(String(after[0]));
      chasers.setReleaseHi(String(after[1]));
      chasers.setMaxTilt(String(info.los_max_tilt_deg ?? THREAT_DEFAULTS.maxTilt));
    } else {
      chasers.setThreats('0');           // no chaser inputs in this checkpoint
    }
  }, [info]);

  const heightMode = !!info && info.obs_position_mode === 'height';
  const recentreDisabled = !form.handoffOn || heightMode;

  function buildForm() {
    const formData = new FormData();
    choice.addToForm(formData);
    addHandoffFields(formData, form, 0);
    if (recentre && !recentreDisabled) formData.append('recentre_z', recentreZ || 20);
    addChaserFields(formData, chasers);
    if (hitRadius) formData.append('hit_radius', hitRadius);
    return formData;
  }

  function runOneFlight() {
    run('Running one episode...', async () => {
      const data = await postForm('/api/two_phase/trajectory', buildForm());
      const { track, log, diag, ...meta } = data;
      setResult({ kind: 'flight', data, track, meta: { controller: 'PID', ...meta }, log, diag });
      setStatus({ kind: 'ok', message: 'Done - press Space to play.' });
    });
  }

  function runBatch() {
    run(`Running ${form.runs || 75} test runs...`, async () => {
      const formData = buildForm();
      formData.append('n_scenarios', form.runs || 75);
      const data = await postForm('/api/two_phase/evaluate', formData);
      setResult({ kind: 'batch', data, distance: form.distance, switchDist: form.switchDist });
      setStatus({ kind: 'ok', message: 'Done.' });
    });
  }

  function applyPreset(key) {
    form.setDistance(String(PRESETS[key].distance));
    form.setSwitchDist(String(PRESETS[key].switchDist));
  }

  return html`
    <div>
      <${CheckpointCard} choice=${choice} note=${info ? describeCheckpoint(info) : ''} />

      <${Card} title="Scenario">
        <${HandoffToggle} form=${form} model="model" />
        <${HandoffFields} form=${form} controller="PID" cruiseMin="0"
          cruiseHint="Use the run's --max-speed. 0 = old slow PID (arrives at ~0.5 m/s).">
          <${NumberField} label="Hit zone radius (m)"
            hint="The flight ends as a Hit the moment the drone is this close to the target. Empty = 0.25 m. The model's zone-timer inputs keep the radius it was trained with."
            value=${hitRadius} onChange=${setHitRadius} placeholder="0.25" step="0.05" min="0.01" />
        <//>
        <${Toggle} checked=${recentre} disabled=${recentreDisabled} onChange=${setRecentre} style=${{ marginTop: '14px' }}
          title="Re-centre the position input at the handoff">
          From the handoff on, the policy sees x,y measured from the handoff point (like training's re-centring); the drone still flies in real coordinates.
          Altitude is capped at${' '}
          <input type="number" value=${recentreZ} disabled=${recentreDisabled} step="1" min="1" onChange=${(e) => setRecentreZ(e.target.value)} />
          m.
        <//>
        <p className="note">${info ? describeRecentre(info) : ''}</p>
        <${Chips} items=${Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label }))} onPick=${applyPreset} />
        <p className="note warn">${budgetHint(form.distance)}</p>
      <//>

      <${ChasersCard} c=${chasers} info=${info} />

      <div className="actions">
        <button className="primary" type="button" disabled=${busy} onClick=${runOneFlight}>Run & play in 3D</button>
        <button className="secondary" type="button" disabled=${busy} onClick=${runBatch}>Run test batch (no video)</button>
      </div>
      <${Status} status=${status} />
      ${result && result.kind === 'flight' && html`<${FlightResult} result=${result} />`}
      ${result && result.kind === 'batch' && html`<${BatchResult} result=${result} distance=${result.distance} switchDist=${result.switchDist} />`}
    </div>`;
}
