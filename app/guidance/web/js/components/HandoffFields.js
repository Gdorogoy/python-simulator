// The scenario fields that the Angle and Acro tabs share. Both tabs fly the same kind of episode:
//   1. a scripted controller flies from `distance` m away down to `switchDist` m from the target,
//   2. then the trained policy takes over (the "handoff") and flies the rest.
// With the handoff switched off the policy flies from the start by itself.
import { html } from '../html.js';
import { NumberField, Toggle } from './ui.js';

// `form` comes from useHandoffForm() (hooks.js). `model` is what the tab calls the trained network ("model" or "policy").
export function HandoffToggle({ form, model }) {
  return html`
    <${Toggle} checked=${form.handoffOn} onChange=${form.setHandoffOn} style=${{ marginBottom: '14px' }}
      title=${`Handoff: the controller flies the spawn leg, then the ${model} takes over`}>
      Off = just the ${model}: it flies from the spawn point to the target by itself (target distance = how far it starts).
    <//>`;
}

// The fields sit in one grid; a tab can append its own extra fields as `children`.
// `controller` names the scripted flyer in the hints ("PID" for angle drones, "controller" for acro);
// `cruiseMin` / `cruiseHint` let a tab word the cruise-speed field for its drone style.
export function HandoffFields({ form, controller, cruiseMin, cruiseHint, children }) {
  const off = !form.handoffOn;       // these fields only matter when a controller flies the first part
  return html`
    <div className="grid">
    <${NumberField} label="Target distance (m)" hint=${`The ${controller} flies it alone down to the switch distance.`}
      value=${form.distance} onChange=${form.setDistance} step="1" min="1" />
    <${NumberField} label="Switch distance (m)" hint="Where the policy takes over."
      value=${form.switchDist} onChange=${form.setSwitchDist} disabled=${off} step="0.5" min="1" />
    <${NumberField} label="Cruise speed (m/s)" hint=${cruiseHint}
      value=${form.cruise} onChange=${form.setCruise} disabled=${off} step="1" min=${cruiseMin} />
    <${NumberField} label="Handoff speed (m/s)" hint="Speed when the policy takes over."
      value=${form.handoffSpeed} onChange=${form.setHandoffSpeed} disabled=${off} placeholder="cruise / 2" step="1" min="0" />
    <${NumberField} label="Sideways speed at handoff (fraction)" hint="Random sideways velocity up to this fraction of the handoff speed (training used 0.3)."
      value=${form.lateral} onChange=${form.setLateral} disabled=${off} step="0.05" min="0" />
    <${NumberField} label="Tilt at handoff (deg)" hint="Roll and pitch random within +-this (training used 15)."
      value=${form.tilt} onChange=${form.setTilt} disabled=${off} step="1" min="0" />
    <${NumberField} label="Test runs" hint='Used by "Run test batch".'
      value=${form.runs} onChange=${form.setRuns} step="1" min="1" />
    <${NumberField} label="Seed (optional)" hint="Same seed = same scenarios."
      value=${form.seed} onChange=${form.setSeed} placeholder="random" />
    ${children}
    </div>`;
}
