// Reusable React hooks. A hook is a function whose name starts with "use" and that can keep state.
import { useCallback, useEffect, useState } from 'react';
import { store } from './storage.js';
import { getJson } from './api.js';

// Like useState, but the value is also saved in localStorage under `key`, so it survives a page reload.
export function useStoredState(key, initialValue) {
  const [value, setValue] = useState(() => store.get(key, initialValue));
  const update = useCallback((next) => {
    setValue(next);
    store.set(key, next);
  }, [key]);
  return [value, update];
}

// Runs a slow server job and keeps the status line up to date.
//   const { status, busy, run, setStatus } = useRunner();
//   run('Running...', async () => { ...await server call...; setStatus({ kind: 'ok', message: 'Done.' }); });
// status.kind is '' | 'busy' | 'ok' | 'error'. While busy, the message gets a running "12 s" counter.
export function useRunner() {
  const [status, setStatus] = useState({ kind: '', message: '' });
  const [busy, setBusy] = useState(false);

  async function run(busyMessage, job) {
    const startedAt = Date.now();
    setBusy(true);
    setStatus({ kind: 'busy', message: busyMessage });
    const timer = setInterval(() => {
      const seconds = Math.round((Date.now() - startedAt) / 1000);
      setStatus({ kind: 'busy', message: `${busyMessage}  ${seconds} s` });
    }, 1000);
    try {
      await job();
    } catch (error) {
      setStatus({ kind: 'error', message: 'Error: ' + (error.message || error) });
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  }

  return { status, busy, run, setStatus };
}

// The list of checkpoint paths the server found (runs/, app/control/, ...). Empty until loaded.
export function useCheckpointList() {
  const [checkpoints, setCheckpoints] = useState([]);
  useEffect(() => {
    getJson('/api/checkpoints')
      .then((data) => setCheckpoints(data.checkpoints))
      .catch(() => setCheckpoints([]));
  }, []);
  return checkpoints;
}

// Everything one tab needs to know about "which checkpoint am I using":
//   source      'pick' (a file the server lists) or 'upload' (a file from your computer)
//   selected    the picked path, file = the uploaded File
//   info        what that run was trained with (its run_config.json), or null
//   addToForm   puts the choice into a FormData (throws a readable error if nothing is chosen)
// `prefix` ('tp' or 'ac') only decides the localStorage keys, so each tab remembers its own choice.
export function useCheckpointChoice(prefix, checkpoints) {
  const [source, setSource] = useStoredState(prefix + 'Source', 'pick');
  const [savedPath, setSavedPath] = useStoredState(prefix + 'Checkpoint', null);
  const [file, setFile] = useState(null);
  const [info, setInfo] = useState(null);

  // the saved choice only counts while it is still in the server's list
  const selected = savedPath && checkpoints.includes(savedPath) ? savedPath : checkpoints[0] || '';

  useEffect(() => {
    let outdated = false;                 // a newer selection started: ignore this answer
    setInfo(null);
    if (!selected) return undefined;
    getJson('/api/checkpoint_info?path=' + encodeURIComponent(selected))
      .then((data) => { if (!outdated && data.has_config) setInfo(data); })
      .catch(() => { /* the info only pre-fills fields, the page works without it */ });
    return () => { outdated = true; };
  }, [selected]);

  function addToForm(formData) {
    if (source === 'upload') {
      if (!file) throw new Error('Choose a .pt file to upload, or switch to "Pick from server".');
      formData.append('file', file);
    } else {
      if (!selected) throw new Error('No checkpoint selected.');
      formData.append('checkpoint', selected);
    }
  }

  return { source, setSource, selected, setSelected: setSavedPath, file, setFile, info, addToForm, list: checkpoints };
}

// The fields the Angle and Acro tabs have in common (the handoff scenario). `keys` are the localStorage keys
// (they differ per tab for historical reasons), `defaults` the starting values. Values stay strings so a
// half-typed number is never rewritten under the user's fingers.
export function useHandoffForm(keys, defaults) {
  const [handoffOn, setHandoffOn] = useStoredState(keys.handoffOn, true);
  const [distance, setDistance] = useStoredState(keys.distance, defaults.distance);
  const [switchDist, setSwitchDist] = useStoredState(keys.switchDist, defaults.switchDist);
  const [cruise, setCruise] = useStoredState(keys.cruise, defaults.cruise);
  const [handoffSpeed, setHandoffSpeed] = useStoredState(keys.handoffSpeed, '');
  const [lateral, setLateral] = useStoredState(keys.lateral, defaults.lateral);
  const [tilt, setTilt] = useStoredState(keys.tilt, defaults.tilt);
  const [runs, setRuns] = useStoredState(keys.runs, '75');
  const [seed, setSeed] = useStoredState(keys.seed, '');
  return {
    handoffOn, setHandoffOn, distance, setDistance, switchDist, setSwitchDist, cruise, setCruise,
    handoffSpeed, setHandoffSpeed, lateral, setLateral, tilt, setTilt, runs, setRuns, seed, setSeed,
  };
}

// Adds the common handoff fields to a FormData, with the names the server expects.
export function addHandoffFields(formData, form, cruiseFallback) {
  formData.append('distance', form.distance);
  formData.append('switch_dist', form.switchDist);
  formData.append('cruise_speed', form.cruise || cruiseFallback);
  formData.append('handoff', form.handoffOn);
  formData.append('lateral_frac', form.lateral || 0);
  formData.append('tilt_deg', form.tilt || 0);
  if (form.handoffSpeed) formData.append('handoff_speed', form.handoffSpeed);
  if (form.seed) formData.append('seed', form.seed);
}
