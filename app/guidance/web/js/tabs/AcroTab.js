// "Acro eval" tab: acro-style drones (the action is thrust + body rates).
// Same page as the Angle tab (shared cards and result views); only the drone-specific fields differ:
// tilt limit, policy rate and launch start here, re-centring on the angle side.
import { useEffect, useState } from 'react';
import { html } from '../html.js';
import { postForm } from '../api.js';
import { addHandoffFields, useCheckpointChoice, useHandoffForm, useRunner, useStoredState } from '../hooks.js';
import { Card, Chips, NumberField, SelectField, Status } from '../components/ui.js';
import { CheckpointCard } from '../components/CheckpointCard.js';
import { HandoffFields, HandoffToggle } from '../components/HandoffFields.js';
import { ChasersCard, useChasers } from '../components/ChasersCard.js';
import { BatchResult, FlightResult } from '../components/EvalResults.js';

const STORAGE_KEYS = {
  handoffOn: 'acHandoffOn', distance: 'acDistance', switchDist: 'acSwitchDist', cruise: 'acCruise',
  handoffSpeed: 'acHandoffSpeed', lateral: 'acLateral', tilt: 'acHoTilt', runs: 'acN', seed: 'acSeed',
};
const DEFAULTS = { distance: '40', switchDist: '4', cruise: '8', lateral: '0.3', tilt: '15' };
const PRESETS = {
  close: { label: 'Close, slow (40 m, switch 4, 8 m/s)', distance: 40, switchDist: 4, cruise: 8 },
  mid: { label: 'Mid (60 m, switch 12, 8 m/s)', distance: 60, switchDist: 12, cruise: 8 },
  far: { label: 'Far, mid speed (80 m, switch 25, 18 m/s)', distance: 80, switchDist: 25, cruise: 18 },
};
const RATE_OPTIONS = [
  { value: '4', label: '60 Hz (4 physics steps per action, as in Isaac)' },
  { value: '1', label: '240 Hz (1 step per action)' },
];
const START_OPTIONS = [
  { value: 'rest', label: 'from rest' },
  { value: 'auto', label: 'as trained (launch speed from run_config)' },
];

// One sentence about what the selected checkpoint was trained with.
function describeCheckpoint(info) {
  const parts = [];
  if (info.distance_high != null) parts.push(`trained for ${info.distance_low}-${info.distance_high} m`);
  if (info.max_speed != null) parts.push(`max speed ${info.max_speed} m/s`);
  if (info.spawn_dist_low != null) parts.push(`spawn leg ${info.spawn_dist_low}-${info.spawn_dist_high} m`);
  if (info.obs_position_mode) parts.push(`position input: ${info.obs_position_mode}`);
  let text = parts.join(', ');
  if (info.distance_high != null) text += ` | switch distance should be <= ${info.distance_high} m`;
  if (info.lateral_frac != null) text += ` | handoff sideways ${info.lateral_frac}, tilt ${info.handoff_tilt_deg} deg (pre-set)`;
  if (info.runner) text += ` | MOVING TARGET: runner ${info.runner_speed_min ?? info.runner_speed}-${info.runner_speed} m/s, escape ${info.runner_escape_dist} m`;
  return text;
}

export function AcroTab({ checkpoints }) {
  const choice = useCheckpointChoice('ac', checkpoints);
  const form = useHandoffForm(STORAGE_KEYS, DEFAULTS);
  const chasers = useChasers('ac');
  const [tiltLimit, setTiltLimit] = useStoredState('acTilt', '120');
  const [policyRate, setPolicyRate] = useStoredState('acRepeat', '4');
  const [start, setStart] = useStoredState('acStart', 'rest');
  const [hitRadius, setHitRadius] = useStoredState('acHitRadius', '');
  // moving target (only used for a --runner run; empty = the run's value)
  const [runnerSpeed, setRunnerSpeed] = useStoredState('acRunnerSpeed', '');
  const [runnerSpeedMin, setRunnerSpeedMin] = useStoredState('acRunnerSpeedMin', '');
  const [runnerEscape, setRunnerEscape] = useStoredState('acRunnerEscape', '');
  const { status, busy, run, setStatus } = useRunner();
  const [result, setResult] = useState(null);

  // When the selected checkpoint's run config arrives, pre-fill the fields it knows about.
  const info = choice.info;
  useEffect(() => {
    if (!info) return;
    if (info.max_speed != null) form.setCruise(String(info.max_speed));
    if (info.spawn_dist_low != null) form.setDistance(String(Math.round((info.spawn_dist_low + info.spawn_dist_high) / 2)));
    if (info.distance_high != null) form.setSwitchDist(String(+((info.distance_low + info.distance_high) / 2).toFixed(1)));
    if (info.lateral_frac != null) {
      form.setLateral(String(info.lateral_frac));
      form.setTilt(String(info.handoff_tilt_deg ?? 15));
    }
    if (info.runner) {
      setRunnerSpeed(info.runner_speed != null ? String(info.runner_speed) : '');
      setRunnerSpeedMin(info.runner_speed_min != null ? String(info.runner_speed_min) : '');
      setRunnerEscape(info.runner_escape_dist != null ? String(info.runner_escape_dist) : '');
    }
  }, [info]);

  function buildForm() {
    const formData = new FormData();
    choice.addToForm(formData);
    addHandoffFields(formData, form, 8);
    formData.append('tilt_limit_deg', tiltLimit || 120);
    formData.append('action_repeat', policyRate);
    formData.append('start', start);
    if (hitRadius) formData.append('hit_radius', hitRadius);
    if (info && info.runner) {
      if (runnerSpeed) formData.append('runner_speed', runnerSpeed);
      if (runnerSpeedMin) formData.append('runner_speed_min', runnerSpeedMin);
      if (runnerEscape) formData.append('runner_escape_dist', runnerEscape);
    }
    return formData;
  }

  function runOneFlight() {
    run('Running one episode...', async () => {
      const data = await postForm('/api/acro/trajectory', buildForm());
      const { track, log, diag, ...meta } = data;
      setResult({ kind: 'flight', data, track, meta, log, diag });
      setStatus({ kind: 'ok', message: 'Done - press Space to play.' });
    });
  }

  function runBatch() {
    run(`Running ${form.runs || 75} test runs (numpy plant, can take a minute)...`, async () => {
      const formData = buildForm();
      formData.append('n_scenarios', form.runs || 75);
      const data = await postForm('/api/acro/evaluate', formData);
      setResult({ kind: 'batch', data, distance: form.distance, switchDist: form.switchDist });
      setStatus({ kind: 'ok', message: 'Done.' });
    });
  }

  function applyPreset(key) {
    form.setDistance(String(PRESETS[key].distance));
    form.setSwitchDist(String(PRESETS[key].switchDist));
    form.setCruise(String(PRESETS[key].cruise));
  }

  return html`
    <div>
      <${CheckpointCard} choice=${choice} note=${info ? describeCheckpoint(info) : ''}>
        <p className="note">Scored like the trainer's eval: deterministic policy, first episode only, per-distance time budget. Runs in the numpy plant, so it also checks that the plant matches Isaac.</p>
      <//>

      <${Card} title="Scenario">
        <${HandoffToggle} form=${form} model="model" />
        <${HandoffFields} form=${form} controller="controller" cruiseMin="1" cruiseHint="Use the run's --max-speed.">
          <${NumberField} label="Hit zone radius (m)"
            hint="The flight ends as a Hit the moment the drone is this close to the target. Empty = the radius the run was trained with (--hit-radius), else 0.25 m."
            value=${hitRadius} onChange=${setHitRadius} placeholder="run's value" step="0.05" min="0.01" />
          <${NumberField} label="Tilt limit (deg)" hint="Acro only: the run's --tilt-limit." value=${tiltLimit} onChange=${setTiltLimit} step="10" />
          <${SelectField} label="Policy rate" value=${policyRate} onChange=${setPolicyRate} options=${RATE_OPTIONS} />
          <${SelectField} label="Start (handoff off)" value=${start} onChange=${setStart} disabled=${form.handoffOn} options=${START_OPTIONS} />
        <//>
        <${Chips} items=${Object.entries(PRESETS).map(([key, p]) => ({ key, label: p.label }))} onPick=${applyPreset} />
      <//>

      ${info && info.runner && html`
        <${Card} title="Moving target (runner)">
          <p className="note">This run trained against a fleeing target: a drone that hovers on the target until the policy takes over, then flies straight away from it at its speed. Inside the hit radius = Hit; once the runner is farther than the escape distance from its own spawn point = Escape (lost), wherever the drone is. Empty = the run's value.</p>
          <${NumberField} label="Runner max speed (m/s)" value=${runnerSpeed} onChange=${setRunnerSpeed} placeholder="run's value" step="1" min="0.5" />
          <${NumberField} label="Runner min speed (m/s)" hint="Each flight draws its speed from min..max." value=${runnerSpeedMin} onChange=${setRunnerSpeedMin} placeholder="run's value" step="1" min="0.5" />
          <${NumberField} label="Escape radius around its spawn point (m)" value=${runnerEscape} onChange=${setRunnerEscape} placeholder="run's value" step="10" min="1" />
        <//>`}

      <${ChasersCard} c=${chasers} info=${info} unavailable=${true} />

      <div className="actions">
        <button className="primary" type="button" disabled=${busy} onClick=${runOneFlight}>Run & play in 3D</button>
        <button className="secondary" type="button" disabled=${busy} onClick=${runBatch}>Run test batch (no video)</button>
      </div>
      <${Status} status=${status} />
      ${result && result.kind === 'flight' && html`<${FlightResult} result=${result} distance=${form.distance} switchDist=${form.switchDist} />`}
      ${result && result.kind === 'batch' && html`<${BatchResult} result=${result} distance=${result.distance} switchDist=${result.switchDist} />`}
    </div>`;
}
