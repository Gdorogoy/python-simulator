// The "Chasing drones" card, shared by the Angle and Acro tabs so both pages look the same.
// The acro env has no chasers yet: that tab passes `unavailable` and the card shows the same fields, switched off.
import { html } from '../html.js';
import { useStoredState } from '../hooks.js';
import { Card, NumberField, SelectField } from './ui.js';

// match the trainer's --threat-* defaults (release = --threat-release-after LO HI, maxTilt = --los-max-tilt-deg)
export const THREAT_DEFAULTS = { count: '0', speed: '50', speedMin: '', spread: '30', spawnMin: '15', spawn: 'uniform', buffer: '0.5',
                                 radius: '', releaseLo: '0.33', releaseHi: '0.99', maxTilt: '90',
                                 wHeadOn: '1', wTail: '1', reaction: '1.5',
                                 tailLo: '15', tailHi: '40', jitter: '2' };

const SPAWN_OPTIONS = [
  { value: 'uniform', label: 'Uniform: random points around the base' },
  { value: 'base', label: 'Base: all from one point' },
  { value: 'geometry', label: 'Geometry: at the handoff, relative to the drone' },
];

// `prefix` only decides the localStorage keys ('tp' for angle keeps the old saved values).
export function useChasers(prefix) {
  const [threats, setThreats] = useStoredState(prefix + 'Threats', THREAT_DEFAULTS.count);
  const [speed, setSpeed] = useStoredState(prefix + 'ThreatSpeed', THREAT_DEFAULTS.speed);
  const [spread, setSpread] = useStoredState(prefix + 'ThreatSpread', THREAT_DEFAULTS.spread);
  const [speedMin, setSpeedMin] = useStoredState(prefix + 'ThreatSpeedMin', THREAT_DEFAULTS.speedMin);
  const [spawnMin, setSpawnMin] = useStoredState(prefix + 'ThreatSpawnMin', THREAT_DEFAULTS.spawnMin);
  const [spawn, setSpawn] = useStoredState(prefix + 'ThreatSpawn', THREAT_DEFAULTS.spawn);
  const [buffer, setBuffer] = useStoredState(prefix + 'ThreatBuffer', THREAT_DEFAULTS.buffer);
  const [radius, setRadius] = useStoredState(prefix + 'ThreatRadius', THREAT_DEFAULTS.radius);
  const [releaseLo, setReleaseLo] = useStoredState(prefix + 'ThreatReleaseLo', THREAT_DEFAULTS.releaseLo);
  const [releaseHi, setReleaseHi] = useStoredState(prefix + 'ThreatReleaseHi', THREAT_DEFAULTS.releaseHi);
  const [maxTilt, setMaxTilt] = useStoredState(prefix + 'ThreatMaxTilt', THREAT_DEFAULTS.maxTilt);
  const [wHeadOn, setWHeadOn] = useStoredState(prefix + 'ThreatWHead', THREAT_DEFAULTS.wHeadOn);
  const [wTail, setWTail] = useStoredState(prefix + 'ThreatWTail', THREAT_DEFAULTS.wTail);
  const [reaction, setReaction] = useStoredState(prefix + 'ThreatReaction', THREAT_DEFAULTS.reaction);
  const [tailLo, setTailLo] = useStoredState(prefix + 'ThreatTailLo', THREAT_DEFAULTS.tailLo);
  const [tailHi, setTailHi] = useStoredState(prefix + 'ThreatTailHi', THREAT_DEFAULTS.tailHi);
  const [jitter, setJitter] = useStoredState(prefix + 'ThreatJitter', THREAT_DEFAULTS.jitter);
  return { wHeadOn, setWHeadOn, wTail, setWTail, reaction, setReaction, tailLo, setTailLo, tailHi, setTailHi, jitter, setJitter,
           threats, setThreats, speed, setSpeed, spread, setSpread, speedMin, setSpeedMin, spawnMin, setSpawnMin,
           spawn, setSpawn, buffer, setBuffer, radius, setRadius, releaseLo, setReleaseLo, releaseHi, setReleaseHi,
           maxTilt, setMaxTilt };
}

export function addChaserFields(formData, c) {
  formData.append('threats', c.threats || 0);
  formData.append('threat_speed', c.speed || THREAT_DEFAULTS.speed);
  formData.append('threat_spread', c.spread || THREAT_DEFAULTS.spread);
  if (c.speedMin) formData.append('threat_speed_min', c.speedMin);
  formData.append('threat_spawn_min', c.spawnMin || THREAT_DEFAULTS.spawnMin);
  formData.append('threat_spawn', c.spawn || THREAT_DEFAULTS.spawn);
  formData.append('threat_buffer', c.buffer || THREAT_DEFAULTS.buffer);
  if (c.radius) formData.append('threat_radius', c.radius);
  formData.append('threat_release_lo', c.releaseLo || THREAT_DEFAULTS.releaseLo);
  formData.append('threat_release_hi', c.releaseHi || THREAT_DEFAULTS.releaseHi);
  formData.append('threat_max_tilt_deg', c.maxTilt || THREAT_DEFAULTS.maxTilt);
  formData.append('threat_w_head_on', c.wHeadOn || THREAT_DEFAULTS.wHeadOn);
  formData.append('threat_w_tail', c.wTail || THREAT_DEFAULTS.wTail);
  formData.append('threat_reaction_s', c.reaction || THREAT_DEFAULTS.reaction);
  formData.append('threat_tail_lo', c.tailLo || THREAT_DEFAULTS.tailLo);
  formData.append('threat_tail_hi', c.tailHi || THREAT_DEFAULTS.tailHi);
  formData.append('threat_jitter', c.jitter || THREAT_DEFAULTS.jitter);
}

export function ChasersCard({ c, info, unavailable }) {
  const off = unavailable || !(parseInt(c.threats, 10) > 0);
  const base = c.spawn === 'base';
  const geo = c.spawn === 'geometry';
  const slots = unavailable
    ? 'The acro env has no chasing drones yet, so these are switched off.'
    : info && info.num_threats ? `This checkpoint has ${info.num_threats} chaser input slot(s).`
      : info ? 'This checkpoint was trained without chaser inputs, so keep this at 0.' : '';
  return html`
    <${Card} title="Chasing drones">
      <p className="note">
        Chasers are drones with the learner's own physics, flown straight at it at their speed cap (line of sight, no other limit
        except the tilt clamp). They hover where they spawn until the model has taken over and the drone has covered their release
        fraction of the distance it had left at the handoff; one that gets within the catch radius ends the flight as "Intercepted".
        The viewer has no contact physics: the collision radius only sets the size they are drawn at.
        ${slots}
      </p>
      <div className="grid">
        <${NumberField} label="Chasing drones" hint="0 = none. Observation is 23 + 7 per slot; cannot exceed the checkpoint's slots."
          value=${c.threats} onChange=${c.setThreats} disabled=${unavailable} step="1" min="0" />
        <${NumberField} label="Chaser speed cap (m/s)" hint="Commanded chase speed, the only limit on the chase."
          value=${c.speed} onChange=${c.setSpeed} disabled=${off} step="1" min="0" />
        <${NumberField} label="Chaser speed, lowest (m/s)" hint="Empty = fixed at the speed cap; otherwise each chaser draws a cap between this and the speed cap."
          value=${c.speedMin} onChange=${c.setSpeedMin} disabled=${off} step="1" min="0" />
        <${SelectField} label="Chaser spawn" hint="Uniform: random points around the base. Base: all from the base, side by side."
          value=${c.spawn} onChange=${c.setSpawn} disabled=${off} options=${SPAWN_OPTIONS} />
        <${NumberField} label="Chaser spawn, farthest (m)" hint="Uniform spawn: outer radius of the ball around the base."
          value=${c.spread} onChange=${c.setSpread} disabled=${off || base || geo} step="1" min="0" />
        <${NumberField} label="Chaser spawn, nearest (m)" hint="Uniform spawn: inner radius (0 = anywhere in the ball)."
          value=${c.spawnMin} onChange=${c.setSpawnMin} disabled=${off || base || geo} step="1" min="0" />
        <${NumberField} label="Chaser spacing buffer (m)" hint="Base spawn: clearance between neighbouring chaser bodies, on top of their size."
          value=${c.buffer} onChange=${c.setBuffer} disabled=${off || !base} step="0.1" min="0" />
        <${NumberField} label="Chaser collision radius (m)" hint="Empty = the drone's own body. Sets the drawn size and the base-spawn spacing."
          value=${c.radius} onChange=${c.setRadius} disabled=${off} step="0.1" min="0" />
        <${NumberField} label="Chaser release, earliest (fraction covered)" hint="Each chaser starts moving once the drone has covered a random fraction (between earliest and latest) of its handoff distance."
          value=${c.releaseLo} onChange=${c.setReleaseLo} disabled=${off || geo} step="0.05" min="0" max="0.99" />
        <${NumberField} label="Chaser release, latest (fraction covered)" hint="0.33 and 0.99 = released between 33% and 99% of the way after the handoff. Equal values = fixed."
          value=${c.releaseHi} onChange=${c.setReleaseHi} disabled=${off || geo} step="0.05" min="0" max="0.99" />
        <${NumberField} label="Chaser tilt clamp (deg)" hint="Airframe limit of the chaser (the trainer's --los-max-tilt-deg), not a chase setting."
          value=${c.maxTilt} onChange=${c.setMaxTilt} disabled=${off} step="5" min="5" max="90" />
        <${NumberField} label="Geometry weight: head-on" hint="Geometry spawn: relative weight of chasers that appear ahead and come straight at the drone."
          value=${c.wHeadOn} onChange=${c.setWHeadOn} disabled=${off || !geo} step="0.5" min="0" />
        <${NumberField} label="Geometry weight: tail" hint="Relative weight of chasers that appear behind the drone and chase it."
          value=${c.wTail} onChange=${c.setWTail} disabled=${off || !geo} step="0.5" min="0" />
        <${NumberField} label="Head-on closing time (s)" hint="Head-on chasers spawn at least this many seconds of closing time ahead (min 20 m)."
          value=${c.reaction} onChange=${c.setReaction} disabled=${off || !geo} step="0.1" min="0.1" />
        <${NumberField} label="Tail gap, nearest (m)" hint="Distance behind the drone of a tail chaser."
          value=${c.tailLo} onChange=${c.setTailLo} disabled=${off || !geo} step="1" min="0" />
        <${NumberField} label="Tail gap, farthest (m)" hint=""
          value=${c.tailHi} onChange=${c.setTailHi} disabled=${off || !geo} step="1" min="0" />
        <${NumberField} label="Head-on jitter (m)" hint="Random offset of a head-on chaser from the drone's line of travel."
          value=${c.jitter} onChange=${c.setJitter} disabled=${off || !geo} step="0.5" min="0" />
      </div>
    <//>`;
}
